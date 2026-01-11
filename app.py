from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import pipeline
from keybert import KeyBERT
import logging
from typing import List, Optional, Dict
import re
from summary import SummaryService, get_supported_languages, normalize_language_code, get_language_name
import yt_dlp
import tempfile
import os
import time
import torch
import json
import asyncio
from threading import Lock
from queue import Queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("Starting TubeScribe API server...")
    yield
    # Shutdown - cleanup SSE connections
    logger.info("Shutting down... cleaning up connections...")
    with progress_lock:
        progress_queues.clear()
        progress_store.clear()
    logger.info("Cleanup complete")

app = FastAPI(title="TubeScribe API", version="1.0.0", lifespan=lifespan)

# Progress tracking for frontend
progress_store: Dict[str, Dict] = {}
progress_lock = Lock()
# SSE event queues for real-time updates
progress_queues: Dict[str, Queue] = {}

# GPU detection
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🔧 Device: {device.upper()}")
if device == "cuda":
    logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Summarization Model Initialization ---
# IMPORTANT: mBART is NOT used for summarization - only for translation
# Summarization uses BART/PEGASUS (English-only models)
# Options:
# - "facebook/bart-large-cnn" (default, English only, fast and accurate)
# - "google/pegasus-xsum" (English, abstractive summaries, longer outputs)
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "facebook/bart-large-cnn")

logger.info(f"Loading summarization model (English-only): {SUMMARIZATION_MODEL}...")
try:
    summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL)
    logger.info(f"✅ Summarization model '{SUMMARIZATION_MODEL}' loaded successfully")
except Exception as e:
    logger.error(f"Failed to load summarization model '{SUMMARIZATION_MODEL}': {e}")
    logger.error("Falling back to BART-Large-CNN...")
    try:
        # Fallback to BART (English-only)
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        logger.info("✅ Fallback summarization model (BART-Large-CNN) loaded successfully")
    except Exception as fallback_error:
        logger.error(f"Failed to load fallback summarization model: {fallback_error}")
        summarizer = None

try:
    # This is the best small multilingual model for keywords
    kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
    logger.info("Keyword extraction model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load keyword model: {e}")
    kw_model = None

# Initialize SummaryService with loaded models
summary_service = SummaryService(summarizer=summarizer, kw_model=kw_model)
if summary_service.is_ready():
    logger.info("✅ SummaryService initialized successfully")
else:
    logger.warning("⚠️  SummaryService initialized but some models are missing")

# --- Transcription Model Initialization ---
logger.info("Loading transcription model...")
try:
    # Using Whisper for automatic speech recognition (multilingual support)
    # Using base model for balance between speed and accuracy
    # Enable GPU if available for faster processing
    transcriber = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        chunk_length_s=30,
        return_timestamps=True,
        device=0 if device == "cuda" else -1  # Use GPU if available
    )
    logger.info(f"✅ Transcription model loaded successfully on {device.upper()}")
except Exception as e:
    logger.error(f"Failed to load transcription model: {e}")
    transcriber = None

class TextData(BaseModel):
    text: str
    # Language for summary output. Accepts:
    # - "AUTO" (default): auto-detect and use source language
    # - Language names: "English", "Chinese", "Spanish", etc.
    # - 2-letter codes: "en", "zh", "es", etc.
    # - mBART codes: "en_XX", "zh_CN", etc.
    lang: str = "AUTO" 

class TranscribeRequest(BaseModel):
    videoUrl: str
    targetLanguage: Optional[str] = None  # For translation, but we'll focus on transcription

class TranscribeResponse(BaseModel):
    transcript: str
    detected_language: Optional[str] = None
    processing_time: Optional[float] = None

class ProgressResponse(BaseModel):
    status: str  # "downloading", "transcribing", "completed", "error"
    progress: float  # 0-100
    message: str
    download_progress: Optional[float] = None
    transcription_progress: Optional[float] = None

class SummarizeResponse(BaseModel):
    summary: str
    keywords: List[str]
    source_language: Optional[str] = None
    target_language: Optional[str] = None

def get_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def download_audio(url: str, video_id_param: Optional[str] = None) -> str:
    """Download audio from YouTube video and return path to audio file"""
    video_id = video_id_param or get_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    
    # Create temporary file for audio
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, f"{video_id}.m4a")
    
    # Progress hook for download tracking
    def progress_hook(d):
        if d['status'] == 'downloading':
            # Calculate progress percentage
            if 'total_bytes' in d:
                percent = (d['downloaded_bytes'] / d['total_bytes']) * 100
                downloaded_mb = d['downloaded_bytes'] / (1024 * 1024)
                total_mb = d['total_bytes'] / (1024 * 1024)
                speed_mb = d.get('speed', 0) / (1024 * 1024) if d.get('speed') else 0
                eta = d.get('eta', 0)
                
                logger.info(
                    f"⬇️  Download: {percent:.1f}% "
                    f"({downloaded_mb:.1f}MB / {total_mb:.1f}MB) "
                    f"@ {speed_mb:.1f}MB/s | ETA: {eta}s"
                )
                
                # Update progress store for frontend (will trigger SSE update)
                # Note: video_id is captured from outer scope
                if video_id:
                    with progress_lock:
                        if video_id not in progress_store:
                            progress_store[video_id] = {
                                "status": "downloading",
                                "progress": 0,
                                "message": "Starting download...",
                                "download_progress": 0,
                                "transcription_progress": 0
                            }
                        progress_store[video_id].update({
                            "status": "downloading",
                            "progress": int(percent * 0.3),  # 0-30% for download
                            "download_progress": int(percent),
                            "message": f"Downloading audio ({percent:.0f}%)"
                        })
                        # Notify SSE clients
                        if video_id in progress_queues:
                            try:
                                progress_queues[video_id].put_nowait(progress_store[video_id].copy())
                            except:
                                pass
            elif '_percent_str' in d:
                # Alternative progress format
                logger.info(f"⬇️  Download: {d.get('_percent_str', 'N/A')} | {d.get('_speed_str', 'N/A')}")
            else:
                # Fallback for unknown size
                downloaded_mb = d.get('downloaded_bytes', 0) / (1024 * 1024)
                logger.info(f"⬇️  Download: {downloaded_mb:.1f}MB downloaded...")
        
        elif d['status'] == 'finished':
            logger.info("✅ Download completed, converting audio format...")
            if video_id:
                with progress_lock:
                    if video_id in progress_store:
                        progress_store[video_id].update({
                            "status": "downloading",
                            "download_progress": 100,
                            "progress": 30,
                            "message": "Download completed"
                        })
                        # Notify SSE clients
                        if video_id in progress_queues:
                            try:
                                progress_queues[video_id].put_nowait(progress_store[video_id].copy())
                            except:
                                pass
        elif d['status'] == 'error':
            logger.error(f"❌ Download error: {d.get('error', 'Unknown error')}")
            if video_id:
                with progress_lock:
                    if video_id in progress_store:
                        progress_store[video_id].update({
                            "status": "error",
                            "message": f"Download error: {d.get('error', 'Unknown error')}"
                        })
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': audio_path.replace('.m4a', '.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False,  # We want to see progress
        'no_warnings': False,
        'progress_hooks': [progress_hook],
        # Fix 403 Forbidden errors
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],  # Try android client first, fallback to web
                'player_skip': ['webpage'],  # Skip webpage player which often gets blocked
            }
        },
        'nocheckcertificate': False,
        'ignoreerrors': False,
        'retries': 3,  # Retry up to 3 times on errors
        'fragment_retries': 3,
        'skip_unavailable_fragments': True,
        'keep_fragments': False,
    }
    
    # Try multiple strategies to avoid 403 errors
    strategies = [
        # Strategy 1: Android client (most reliable, bypasses many restrictions)
        {
            **ydl_opts,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android'],
                    'player_skip': ['webpage', 'configs'],
                }
            }
        },
        # Strategy 2: iOS client
        {
            **ydl_opts,
            'extractor_args': {
                'youtube': {
                    'player_client': ['ios'],
                    'player_skip': ['webpage'],
                }
            }
        },
        # Strategy 3: Web client as last resort
        {
            **ydl_opts,
            'extractor_args': {
                'youtube': {
                    'player_client': ['web'],
                }
            }
        }
    ]
    
    last_error = None
    for attempt, opts in enumerate(strategies, 1):
        try:
            logger.info(f"🔗 Attempt {attempt}/{len(strategies)}: Fetching video information...")
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                duration = info.get('duration', 0)
                if duration:
                    minutes = duration // 60
                    seconds = duration % 60
                    logger.info(f"📹 Video info: {info.get('title', 'Unknown')[:60]}...")
                    logger.info(f"⏱️  Duration: {minutes}:{seconds:02d} ({duration}s)")
                
                logger.info("⬇️  Starting audio download...")
                ydl.download([url])
                
                # Find the downloaded file
                wav_path = audio_path.replace('.m4a', '.wav')
                if os.path.exists(wav_path):
                    file_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
                    logger.info(f"✅ Audio file ready: {file_size_mb:.1f}MB")
                    return wav_path
                # Try to find any audio file
                for ext in ['.wav', '.m4a', '.webm', '.mp3']:
                    candidate = audio_path.replace('.m4a', ext)
                    if os.path.exists(candidate):
                        file_size_mb = os.path.getsize(candidate) / (1024 * 1024)
                        logger.info(f"✅ Audio file ready: {file_size_mb:.1f}MB")
                        return candidate
                raise FileNotFoundError("Audio file not found after download")
                
        except yt_dlp.utils.DownloadError as e:
            last_error = e
            error_msg = str(e)
            if "HTTP Error 403" in error_msg or "Forbidden" in error_msg or "403" in error_msg:
                if attempt < len(strategies):
                    logger.warning(f"⚠️  Attempt {attempt} failed with 403, trying next strategy...")
                    continue
                else:
                    logger.error(f"❌ All {len(strategies)} strategies failed with 403 error")
                    raise ValueError(f"Failed to download video after {len(strategies)} attempts. YouTube may be blocking the request. Try updating yt-dlp: pip install -U yt-dlp. Error: {error_msg}")
            else:
                # Other errors, raise immediately
                raise
        except Exception as e:
            last_error = e
            if attempt < len(strategies):
                logger.warning(f"⚠️  Attempt {attempt} failed: {e}, trying next strategy...")
                continue
            else:
                raise
    
    # If we get here, all strategies failed
    if last_error:
        raise last_error
    raise Exception("Download failed for unknown reason")

@app.get("/")
async def root():
    return {
        "service": "TubeScribe API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": {
            "transcriber": transcriber is not None,
            "summarizer": summarizer is not None,
            "keyword_extractor": kw_model is not None,
            "summary_service": summary_service.is_ready()
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_ready": {
            "transcriber": transcriber is not None,
            "summarizer": summarizer is not None,
            "keyword_extractor": kw_model is not None,
            "summary_service": summary_service.is_ready()
        }
    }

@app.get("/api/languages")
async def get_languages():
    """
    Get list of supported languages for summary output.
    
    Returns:
        List of available languages with their codes and names.
    """
    return {
        "languages": get_supported_languages(),
        "default": "AUTO"
    }

@app.get("/api/progress/{video_id}", response_model=ProgressResponse)
async def get_progress(video_id: str):
    """Get transcription progress for a video (legacy polling endpoint)"""
    with progress_lock:
        progress = progress_store.get(video_id, {
            "status": "unknown",
            "progress": 0,
            "message": "No progress data available"
        })
    return ProgressResponse(**progress)

@app.get("/api/progress/{video_id}/stream")
async def stream_progress(video_id: str):
    """
    Server-Sent Events (SSE) stream for real-time progress updates.
    Client connects and receives updates as they happen.
    """
    async def event_generator():
        # Create a queue for this client
        queue = Queue()
        with progress_lock:
            if video_id not in progress_queues:
                progress_queues[video_id] = queue
            else:
                queue = progress_queues[video_id]
        
        # Send initial progress if available
        with progress_lock:
            if video_id in progress_store:
                initial_progress = progress_store[video_id]
                yield f"data: {json.dumps(initial_progress)}\n\n"
                last_progress = initial_progress.copy()
            else:
                last_progress = None
        
        # Keep connection alive and send updates
        last_sent_time = time.time()
        try:
            while True:
                # Check for new progress updates
                try:
                    # Non-blocking check for new progress
                    with progress_lock:
                        if video_id in progress_store:
                            current_progress = progress_store[video_id]
                            
                            # Check if progress changed (compare as dict for better detection)
                            progress_changed = (
                                last_progress is None or
                                current_progress.get("progress") != last_progress.get("progress") or
                                current_progress.get("status") != last_progress.get("status") or
                                current_progress.get("message") != last_progress.get("message")
                            )
                            
                            if progress_changed:
                                yield f"data: {json.dumps(current_progress)}\n\n"
                                last_progress = current_progress.copy()
                                last_sent_time = time.time()
                                
                                # Close stream if completed
                                if current_progress.get("status") in ["completed", "error"]:
                                    break
                            elif time.time() - last_sent_time > 5:
                                # Send keepalive every 5 seconds to prevent timeout
                                yield f": keepalive\n\n"
                                last_sent_time = time.time()
                    
                    # Send keepalive every 5 seconds if no updates
                    await asyncio.sleep(0.05)  # 50ms check interval for faster sync
                    
                except asyncio.CancelledError:
                    # Connection closed by client or server shutdown
                    break
                except Exception as e:
                    logger.error(f"Error in progress stream: {e}")
                    try:
                        yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
                    except:
                        pass  # Client may have disconnected
                    break
                    
        except asyncio.CancelledError:
            # Normal shutdown - connection cancelled
            pass
        except GeneratorExit:
            # Client disconnected
            pass
        except Exception as e:
            logger.warning(f"Unexpected error in SSE stream: {e}")
        finally:
            # Cleanup
            try:
                with progress_lock:
                    if video_id in progress_queues:
                        # Only delete if this is the only queue
                        if progress_queues[video_id] == queue:
                            del progress_queues[video_id]
            except Exception as e:
                logger.warning(f"Error during SSE cleanup: {e}")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe_video(request: TranscribeRequest, background_tasks: BackgroundTasks):
    """
    Transcribe YouTube video using local Whisper model.
    
    Args:
        request: TranscribeRequest containing videoUrl
        
    Returns:
        TranscribeResponse with transcript text including timestamps
    """
    if not transcriber:
        raise HTTPException(
            status_code=503,
            detail="Transcription model not loaded. Please check server logs."
        )
    
    video_id = get_video_id(request.videoUrl)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    try:
        logger.info("=" * 60)
        logger.info(f"🎬 NEW TRANSCRIPTION REQUEST")
        logger.info(f"📹 Video URL: {request.videoUrl}")
        logger.info("=" * 60)
        
        # Helper to update progress and notify SSE clients
        def update_progress(updates: Dict):
            """Helper to update progress and notify SSE clients"""
            with progress_lock:
                if video_id not in progress_store:
                    progress_store[video_id] = {
                        "status": "downloading",
                        "progress": 0,
                        "message": "Starting...",
                        "download_progress": 0,
                        "transcription_progress": 0
                    }
                progress_store[video_id].update(updates)
                # Notify any SSE clients - they'll pick up changes on next check
        
        update_progress({
            "status": "downloading",
            "progress": 0,
            "message": "Preparing to download audio...",
            "download_progress": 0,
            "transcription_progress": 0
        })
        
        # Download audio
        logger.info("📥 Step 1/3: Downloading audio from YouTube...")
        download_start = time.time()
        audio_path = download_audio(request.videoUrl, video_id)  # Pass video_id for progress
        download_time = time.time() - download_start
        logger.info(f"✅ Audio downloaded in {download_time:.1f}s: {audio_path}")
        
        # Update progress after download
        update_progress({
            "status": "transcribing",
            "progress": 30,  # 30% after download
            "message": "Audio ready, starting transcription...",
            "download_progress": 100,
            "transcription_progress": 0
        })
        
        try:
            # Get audio file info for progress estimation
            audio_file_size = os.path.getsize(audio_path)
            # Rough estimate: WAV at 16kHz, 16-bit mono = ~32KB/second
            audio_duration_estimate = audio_file_size / (32 * 1024)
            
            logger.info("🎤 Step 2/3: Transcribing audio with Whisper model...")
            logger.info(f"📊 Audio file size: {audio_file_size / (1024*1024):.1f}MB")
            logger.info(f"⏱️  Estimated duration: ~{audio_duration_estimate:.0f} seconds")
            logger.info(f"⏳ Estimated processing time: ~{audio_duration_estimate * 1.5:.0f} seconds (1-2x real-time on CPU)")
            logger.info("🔄 Processing... (This may take a while, please wait)")
            
            # Enable automatic language detection
            transcribe_start = time.time()
            
            # Start transcription - update progress to indicate we're processing
            update_progress({
                "status": "transcribing",
                "progress": 35,  # Slightly above download completion
                "transcription_progress": 0,
                "message": "Processing audio with AI model..."
            })
            
            # Transcribe with language detection enabled
            # Note: Whisper doesn't provide progress callbacks, so we estimate based on time
            result = transcriber(
                audio_path,
                return_timestamps=True,
                generate_kwargs={"language": None}  # Auto-detect language
            )
            transcribe_time = time.time() - transcribe_start
            
            # Update progress after transcription completes
            update_progress({
                "status": "transcribing",
                "progress": 85,
                "transcription_progress": 100,
                "message": "Finalizing transcript..."
            })
            
            processing_rate = audio_duration_estimate / transcribe_time if transcribe_time > 0 and audio_duration_estimate > 0 else 0
            logger.info(f"✅ Transcription completed in {transcribe_time:.1f}s")
            if processing_rate > 0:
                logger.info(f"⚡ Processing speed: {processing_rate:.2f}x real-time")
            
            # Get detected language if available
            detected_language = result.get('language', None) if isinstance(result, dict) else None
            
            # Format transcript with improved timestamps
            transcript_parts = []
            if isinstance(result, dict) and 'chunks' in result:
                # Whisper returns chunks with timestamps
                for chunk in result['chunks']:
                    timestamp = chunk.get('timestamp', [0, 0])
                    start_time = float(timestamp[0]) if isinstance(timestamp, (list, tuple)) and len(timestamp) > 0 else 0.0
                    
                    # Format timestamp: [HH:MM:SS] or [MM:SS] if less than 1 hour
                    hours = int(start_time // 3600)
                    minutes = int((start_time % 3600) // 60)
                    seconds = int(start_time % 60)
                    milliseconds = int((start_time % 1) * 1000) // 100  # Tenths of seconds
                    
                    if hours > 0:
                        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds}"
                    else:
                        time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds}"
                    
                    text = chunk.get('text', '').strip()
                    if text:
                        transcript_parts.append(f"[{time_str}] {text}")
            elif isinstance(result, dict) and 'text' in result:
                # Simple text result without timestamps
                text = result['text']
                # Try to split by sentences and estimate timestamps
                sentences = text.split('. ')
                total_chars = len(text)
                if total_chars > 0:
                    for i, sentence in enumerate(sentences):
                        if sentence.strip():
                            # Estimate timestamp based on position
                            estimated_time = int((i / len(sentences)) * 300)  # Assume 5 min video
                            minutes = estimated_time // 60
                            seconds = estimated_time % 60
                            transcript_parts.append(f"[{minutes:02d}:{seconds:02d}] {sentence.strip()}.")
                else:
                    transcript_parts.append(text)
            else:
                # Fallback
                transcript_parts.append(str(result))
            
            transcript = "\n".join(transcript_parts)
            
            logger.info("📝 Step 3/3: Formatting transcript...")
            if detected_language:
                logger.info(f"🌍 Detected language: {detected_language}")
            logger.info(f"✅ Transcription completed: {len(transcript)} characters, {len(transcript_parts)} segments")
            logger.info("=" * 60)
            logger.info("✨ TRANSCRIPTION SUCCESSFUL")
            logger.info("=" * 60)
            
            # Update progress to completed
            update_progress({
                "status": "completed",
                "progress": 100,
                "message": "Transcription completed",
                "download_progress": 100,
                "transcription_progress": 100
            })
            
            return TranscribeResponse(
                transcript=transcript,
                detected_language=detected_language,
                processing_time=transcribe_time
            )
            
        finally:
            # Clean up temporary audio file
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    logger.info(f"Cleaned up temporary file: {audio_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error transcribing video: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_multilingual(data: TextData):
    """
    Summarize text and extract keywords using multilingual models.
    
    The 'lang' field accepts flexible language input:
    - "AUTO" (default): auto-detect and output in source language
    - Language names: "English", "Chinese", "Spanish", "French", etc.
    - 2-letter codes: "en", "zh", "es", "fr", etc.
    - mBART codes: "en_XX", "zh_CN", "es_XX", etc.
    
    Args:
        data: TextData containing text and optional language
        
    Returns:
        SummarizeResponse with summary text and keywords list
    """
    if not data.text or not data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if not summary_service.is_ready():
        raise HTTPException(
            status_code=503, 
            detail="Summary service not ready. Please check server logs."
        )
    
    try:
        # Log the language request
        normalized_lang = normalize_language_code(data.lang)
        logger.info(f"📝 Summary request - Language: '{data.lang}' -> '{normalized_lang}'")
        
        # Use the SummaryService to process the text
        result = summary_service.summarize_with_keywords(
            text=data.text,
            language=data.lang,  # Pass user's language choice
            top_n_keywords=10
        )
        
        logger.info(f"✅ Summary generated in {result.get('target_language', 'unknown')} ({len(result['summary'].split())} words)")
        
        return SummarizeResponse(
            summary=result['summary'],
            keywords=result['keywords'],
            source_language=result.get('source_language'),
            target_language=result.get('target_language')
        )
        
    except ValueError as e:
        # Handle validation errors from SummaryService
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in summarize: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")