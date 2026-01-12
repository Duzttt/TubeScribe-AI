"""
Summary Module for TubeScribe AI

Provides text summarization and keyword extraction functionality.
Uses BART/PEGASUS for summarization (English-only) with mBART for translation.
KeyBERT is used for keyword extraction.
"""

import logging
import re
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any, Callable
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast
from keybert import KeyBERT
import torch

# Try to import language detection (optional dependency)
try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    LangDetectException = Exception

logger = logging.getLogger(__name__)


@dataclass
class PartSummary:
    """Represents a summary of a single part/chunk of text."""
    part_number: int
    title: str
    summary: str
    word_count: int
    start_position: int  # Character position in original text
    end_position: int
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "part_number": self.part_number,
            "title": self.title,
            "summary": self.summary,
            "word_count": self.word_count,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "keywords": self.keywords
        }


@dataclass
class StructuredSummary:
    """Represents a complete structured summary with parts and overview."""
    parts: List[PartSummary]
    overview: str
    total_parts: int
    original_word_count: int
    summary_word_count: int
    source_language: str
    target_language: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "parts": [p.to_dict() for p in self.parts],
            "overview": self.overview,
            "total_parts": self.total_parts,
            "original_word_count": self.original_word_count,
            "summary_word_count": self.summary_word_count,
            "source_language": self.source_language,
            "target_language": self.target_language
        }
    
    def get_combined_summary(self, include_part_headers: bool = True) -> str:
        """Get all part summaries combined into a single string."""
        if not self.parts:
            return self.overview
        
        if include_part_headers:
            parts_text = []
            for part in self.parts:
                parts_text.append(f"**{part.title}**\n{part.summary}")
            return "\n\n".join(parts_text) + f"\n\n**Overview**\n{self.overview}"
        else:
            return " ".join([p.summary for p in self.parts])

# mBART language code mapping
# mBART language code mapping (2-letter code -> mBART code)
LANG_CODE_MAP = {
    "en": "en_XX", "es": "es_XX", "fr": "fr_XX", "de": "de_DE",
    "zh": "zh_CN", "ja": "ja_XX", "ko": "ko_KR", "hi": "hi_IN",
    "ar": "ar_AR", "pt": "pt_XX", "it": "it_IT", "ru": "ru_RU",
    "nl": "nl_XX", "tr": "tr_TR", "vi": "vi_VN", "ms": "ms_XX"
}

# Reverse mapping (mBART code -> 2-letter code)
MBART_TO_LANG = {v: k for k, v in LANG_CODE_MAP.items()}

# Human-readable language names
LANG_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "hi": "Hindi",
    "ar": "Arabic", "pt": "Portuguese", "it": "Italian", "ru": "Russian",
    "nl": "Dutch", "tr": "Turkish", "vi": "Vietnamese", "ms": "Malay"
}

# Reverse mapping (name -> 2-letter code)
LANG_NAME_TO_CODE = {v.lower(): k for k, v in LANG_NAMES.items()}


def get_supported_languages() -> List[Dict[str, str]]:
    """
    Get list of supported languages for summary output.
    
    Returns:
        List of dicts with 'code', 'mbart_code', and 'name' for each language.
        Also includes 'AUTO' option for automatic language detection.
    """
    languages = [
        {"code": "AUTO", "mbart_code": "AUTO", "name": "Auto-detect (same as source)"}
    ]
    
    for code, name in LANG_NAMES.items():
        languages.append({
            "code": code,
            "mbart_code": LANG_CODE_MAP[code],
            "name": name
        })
    
    return languages


def normalize_language_code(lang_input: str) -> str:
    """
    Normalize any language input to mBART code.
    
    Accepts:
    - mBART codes: "en_XX", "zh_CN", etc.
    - 2-letter codes: "en", "zh", "es", etc.
    - Language names: "English", "Chinese", "Spanish", etc.
    - "AUTO" for automatic detection
    
    Args:
        lang_input: Language identifier in any supported format.
        
    Returns:
        mBART language code (e.g., "en_XX") or "AUTO".
    """
    if not lang_input:
        return "AUTO"
    
    lang_input = lang_input.strip()
    
    # Already "AUTO"
    if lang_input.upper() == "AUTO":
        return "AUTO"
    
    # Already an mBART code
    if lang_input in MBART_TO_LANG:
        return lang_input
    
    # 2-letter code
    lang_lower = lang_input.lower()
    if lang_lower in LANG_CODE_MAP:
        return LANG_CODE_MAP[lang_lower]
    
    # Language name (case-insensitive)
    if lang_lower in LANG_NAME_TO_CODE:
        code = LANG_NAME_TO_CODE[lang_lower]
        return LANG_CODE_MAP[code]
    
    # Partial name match (e.g., "chin" matches "Chinese")
    for name, code in LANG_NAME_TO_CODE.items():
        if lang_lower in name or name in lang_lower:
            return LANG_CODE_MAP[code]
    
    logger.warning(f"Unknown language '{lang_input}', defaulting to AUTO")
    return "AUTO"


def get_language_name(lang_code: str) -> str:
    """
    Get human-readable language name from any code format.
    
    Args:
        lang_code: Language code in any format.
        
    Returns:
        Human-readable language name.
    """
    if lang_code == "AUTO":
        return "Auto-detect"
    
    # Convert mBART code to 2-letter
    if lang_code in MBART_TO_LANG:
        lang_code = MBART_TO_LANG[lang_code]
    
    return LANG_NAMES.get(lang_code, lang_code)


class SummaryService:
    """
    Service class for text summarization and keyword extraction.
    
    Uses English-only summarization models (BART/PEGASUS) with translation
    preprocessing and postprocessing. Summarization always runs in English.
    """
    
    def __init__(
        self,
        summarizer: Optional[pipeline] = None,
        kw_model: Optional[KeyBERT] = None,
        default_chunk_size: int = 500,
        min_words_for_summary: int = 50,
        default_keyword_count: int = 10,
        translation_model_path: Optional[str] = None
    ):
        """
        Initialize the SummaryService.
        
        Args:
            summarizer: Optional pre-loaded summarization pipeline.
                       If None, will need to be loaded separately.
            kw_model: Optional pre-loaded KeyBERT model.
                     If None, will need to be loaded separately.
            default_chunk_size: Default chunk size for text splitting (words).
            min_words_for_summary: Minimum words required to generate summary.
            default_keyword_count: Default number of keywords to extract.
            translation_model_path: Optional local path or HuggingFace model ID for translation model.
                                   If None, uses default mBART-50 model.
        """
        self.summarizer = summarizer
        self.kw_model = kw_model
        self.default_chunk_size = default_chunk_size
        self.min_words_for_summary = min_words_for_summary
        self.default_keyword_count = default_keyword_count
        
        # Translation model configuration
        self.translation_model_path = translation_model_path or os.getenv(
            "TRANSLATION_MODEL", 
            "facebook/mbart-large-50-many-to-many-mmt"
        )
        
        # Initialize translation components (lazy loading)
        self._translator_model = None
        self._translator_tokenizer = None
        self._translation_device = None
    
    def is_ready(self) -> bool:
        """
        Check if the service has both models loaded.
        
        Returns:
            True if both models are loaded, False otherwise.
        """
        return self.summarizer is not None and self.kw_model is not None
    
    def extract_keywords(
        self,
        text: str,
        top_n: Optional[int] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 2),
        use_mmr: bool = True,
        diversity: float = 0.7
    ) -> List[str]:
        """
        Extract keywords from text using KeyBERT.
        
        Args:
            text: Input text to extract keywords from.
            top_n: Number of keywords to extract (default: self.default_keyword_count).
            keyphrase_ngram_range: Range of n-grams to consider (e.g., (1, 2) for unigrams and bigrams).
            use_mmr: Whether to use Maximal Marginal Relevance for diversity.
            diversity: Diversity parameter for MMR (0.0-1.0, higher = more diverse).
            
        Returns:
            List of extracted keywords/phrases.
            
        Raises:
            ValueError: If kw_model is not loaded or text is empty.
        """
        if not self.kw_model:
            raise ValueError("Keyword extraction model not loaded")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        top_n = top_n or self.default_keyword_count
        
        try:
            keywords_raw = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=keyphrase_ngram_range,
                use_mmr=use_mmr,
                diversity=diversity,
                top_n=top_n
            )
            
            # Extract keyword strings from tuples
            keywords = [kw[0] for kw in keywords_raw if kw and len(kw) > 0]
            
            logger.info(f"Extracted {len(keywords)} keywords")
            return keywords
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            raise
    
    def _clean_transcript_text(self, text: str) -> str:
        """
        Clean transcript text by removing timestamps and formatting.
        
        Removes patterns like:
        - [HH:MM:SS] or [MM:SS] or [HH:MM:SS.mmm] (timestamp patterns)
        - [00:00.0] style timestamps
        
        Args:
            text: Input transcript text with timestamps.
            
        Returns:
            Cleaned text without timestamps.
        """
        # Remove timestamp patterns: [HH:MM:SS], [MM:SS], [HH:MM:SS.mmm], [MM:SS.mmm]
        # Patterns: [00:00], [00:00:00], [00:00.0], [00:00:00.0]
        timestamp_pattern = r'\[\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\]\s*'
        cleaned = re.sub(timestamp_pattern, '', text)
        
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        Fixes false positives for Balkan languages (hr, sl, sr, bs) when text is actually English.
        
        Args:
            text: Input text to detect language for.
            
        Returns:
            Language code (e.g., 'en', 'it', 'es') or 'en' as default.
        """
        if not HAS_LANGDETECT:
            logger.warning("langdetect not available, assuming English")
            return 'en'
        
        try:
            # Use a sample of the text for detection (first 500 chars for speed)
            sample = text[:500] if len(text) > 500 else text
            detected = detect(sample)
            
            # Fix: Override Balkan language false-detection if text contains >70% ASCII
            # This handles cases where langdetect incorrectly identifies English as hr/sl/sr/bs
            balkan_languages = {'hr', 'sl', 'sr', 'bs'}  # Croatian, Slovenian, Serbian, Bosnian
            if detected in balkan_languages:
                # Calculate ASCII percentage
                ascii_chars = sum(1 for c in sample if ord(c) < 128)
                ascii_percentage = (ascii_chars / len(sample)) * 100 if len(sample) > 0 else 0
                
                if ascii_percentage > 70:
                    logger.info(f"🔍 Override: langdetect said {detected}, but >70% ASCII ({ascii_percentage:.1f}%), assuming English")
                    detected = 'en'
            
            detected_name = LANG_NAMES.get(detected, detected)
            sample_preview = sample[:100].replace('\n', ' ')
            logger.info(f"🔍 Language detected: {detected_name} ({detected}) | Sample: {sample_preview}...")
            return detected
        except LangDetectException:
            logger.warning("Language detection failed, assuming English")
            return 'en'
    
    def _get_translator(self):
        """
        Lazy load the translation model and tokenizer.
        Uses mBART-50 for multilingual translation - 100% LOCAL, NO API KEYS NEEDED.
        Automatically uses GPU if available for fast translation.
        Supports local model paths or HuggingFace model IDs.
        
        This is a fully local solution - no external APIs (Gemini, Google Translate, etc.) required.
        """
        if self._translator_model is None:
            try:
                # Detect device (GPU if available, else CPU)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._translation_device = device
                
                model_name = self.translation_model_path
                is_local_path = os.path.exists(model_name) if model_name else False
                
                if is_local_path:
                    logger.info(f"Loading translation model from local path on {device.upper()}: {model_name}")
                else:
                    logger.info(f"Loading translation model (HuggingFace) on {device.upper()}: {model_name}")
                
                # Load tokenizer and model
                # Check if it's a local path or HuggingFace ID
                if is_local_path:
                    # Load from local path
                    self._translator_tokenizer = MBart50TokenizerFast.from_pretrained(
                        model_name,
                        local_files_only=True
                    )
                    self._translator_model = MBartForConditionalGeneration.from_pretrained(
                        model_name,
                        local_files_only=True
                    )
                else:
                    # Load from HuggingFace (will download if not cached)
                    self._translator_tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
                    self._translator_model = MBartForConditionalGeneration.from_pretrained(model_name)
                
                # Move model to GPU if available and optimize
                if device == "cuda":
                    # Use half precision (FP16) for better GPU performance and memory usage
                    try:
                        self._translator_model = self._translator_model.to(device)
                        # Try to use half precision if supported
                        if torch.cuda.is_available() and hasattr(torch.cuda, 'get_device_capability'):
                            device_capability = torch.cuda.get_device_capability(device)
                            if device_capability[0] >= 7:  # Compute capability 7.0+ supports Tensor Cores
                                self._translator_model = self._translator_model.half()
                                logger.info(f"✅ Translation model loaded on GPU with FP16 precision")
                            else:
                                logger.info(f"✅ Translation model loaded successfully on GPU")
                        else:
                            logger.info(f"✅ Translation model loaded successfully on GPU")
                    except Exception as fp16_error:
                        logger.warning(f"Could not use FP16 precision: {fp16_error}, using FP32")
                        self._translator_model = self._translator_model.to(device)
                        logger.info(f"✅ Translation model loaded successfully on GPU (FP32)")
                else:
                    logger.info(f"✅ Translation model loaded successfully on CPU")
                
                # Enable evaluation mode for faster inference
                self._translator_model.eval()
                
                # Log supported languages from tokenizer
                supported_lang_codes = list(self._translator_tokenizer.lang_code_to_id.keys())
                logger.info(f"📋 Translation model supports {len(supported_lang_codes)} languages")
                logger.debug(f"   Supported language codes: {', '.join(sorted(supported_lang_codes)[:20])}...")
                
                # Check if Malay is supported
                malay_variants = ['ms_XX', 'ms_MY', 'ms']
                malay_supported = any(variant in supported_lang_codes for variant in malay_variants)
                if malay_supported:
                    actual_malay_code = next((v for v in malay_variants if v in supported_lang_codes), None)
                    logger.info(f"✅ Malay language support detected: {actual_malay_code}")
                else:
                    logger.warning(f"⚠️  Malay (ms_XX) not found in supported languages. Available codes containing 'ms': {[c for c in supported_lang_codes if 'ms' in c.lower()]}")
                
                # Log model info
                if device == "cuda":
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logger.info(f"   🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                
            except Exception as e:
                logger.error(f"Failed to load translation model: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None, None
        return self._translator_model, self._translator_tokenizer
    
    def _translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> str:
        """
        Translate text from source language to target language using LOCAL mBART-50 model.
        This is a fully local translation - NO API KEYS REQUIRED.
        Automatically uses GPU for fast processing if available.
        
        Args:
            text: Text to translate.
            source_lang: Source language code (2-letter, e.g., 'it', 'en').
            target_lang: Target language code (2-letter, e.g., 'en', 'es').
            progress_callback: Optional callback function(progress, message) to report progress (0-100).
            
        Returns:
            Translated text, or original text if translation fails.
        """
        if source_lang == target_lang:
            if progress_callback:
                progress_callback(100, "No translation needed (same language)")
            return text
        
        # Get mBART language codes
        src_mbart = LANG_CODE_MAP.get(source_lang)
        tgt_mbart = LANG_CODE_MAP.get(target_lang)
        
        if not src_mbart or not tgt_mbart:
            logger.warning(f"Unsupported language pair: {source_lang} -> {target_lang}")
            return text
        
        model, tokenizer = self._get_translator()
        if model is None or tokenizer is None:
            logger.warning("Translation model not available, returning original text")
            return text
        
        # Validate that the language codes are supported by the tokenizer
        if src_mbart not in tokenizer.lang_code_to_id:
            supported_sample = list(tokenizer.lang_code_to_id.keys())[:15]
            logger.error(f"❌ Source language code '{src_mbart}' not supported by translation model.")
            logger.error(f"   Supported codes (sample): {', '.join(sorted(supported_sample))}...")
            logger.warning(f"⚠️  Returning original text (translation failed for {source_lang} -> {target_lang})")
            # Try to find similar language codes
            similar_codes = [code for code in tokenizer.lang_code_to_id.keys() if source_lang.lower() in code.lower() or code.lower().startswith(source_lang.lower())]
            if similar_codes:
                logger.info(f"💡 Similar language codes found: {similar_codes}")
            return text
        
        if tgt_mbart not in tokenizer.lang_code_to_id:
            supported_sample = list(tokenizer.lang_code_to_id.keys())[:15]
            logger.error(f"❌ Target language code '{tgt_mbart}' not supported by translation model.")
            logger.error(f"   Supported codes (sample): {', '.join(sorted(supported_sample))}...")
            logger.warning(f"⚠️  Returning original text (translation failed for {source_lang} -> {target_lang})")
            # Try to find similar language codes
            similar_codes = [code for code in tokenizer.lang_code_to_id.keys() if target_lang.lower() in code.lower() or code.lower().startswith(target_lang.lower())]
            if similar_codes:
                logger.info(f"💡 Similar language codes found: {similar_codes}")
            # Check for Indonesian as potential fallback for Malay (they are related languages)
            if target_lang == "ms" and "id_ID" in tokenizer.lang_code_to_id:
                logger.info(f"💡 Note: Indonesian (id_ID) is available and may be used as a fallback for Malay")
            return text
        
        try:
            # Use the device from model initialization
            device = self._translation_device or ("cuda" if torch.cuda.is_available() else "cpu")
            device_label = "GPU" if device == "cuda" else "CPU"
            
            # Chunk text for large translations to enable progress tracking
            # mBART has max length of 1024 tokens, so we chunk at ~400 words to be safe
            CHUNK_SIZE_WORDS = 400
            text_words = text.split()
            total_words = len(text_words)
            
            # If text is short enough, translate in one go
            if total_words <= CHUNK_SIZE_WORDS:
                if progress_callback:
                    progress_callback(10, "Starting translation...")
                
                # Set source language for tokenizer
                tokenizer.src_lang = src_mbart
                
                # Tokenize input and move to device
                encoded = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True)
                encoded = {k: v.to(device) for k, v in encoded.items()}
                
                # Get target language token ID
                tgt_lang_token_id = tokenizer.lang_code_to_id[tgt_mbart]
                
                if progress_callback:
                    progress_callback(50, f"Translating on {device_label}...")
                
                # Generate translation with forced target language
                # Use torch.no_grad() for faster inference and lower memory usage
                # This is 100% LOCAL translation using GPU (no API calls)
                logger.debug(f"🔄 Translating on {device_label} using local mBART-50 model...")
                
                with torch.no_grad():
                    generated_tokens = model.generate(
                        **encoded,
                        forced_bos_token_id=tgt_lang_token_id,
                        max_length=1024,
                        num_beams=5,  # Beam search for better quality
                        early_stopping=True,
                        do_sample=False
                    )
                
                # Decode translation (move back to CPU for decoding)
                translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                
                if progress_callback:
                    progress_callback(100, "Translation completed")
                
                logger.info(f"✅ Translated on {device_label} from {LANG_NAMES.get(source_lang, source_lang)} to {LANG_NAMES.get(target_lang, target_lang)} ({len(translation.split())} words)")
                return translation
            else:
                # Chunk text and translate in parts for large texts
                if progress_callback:
                    progress_callback(5, f"Preparing to translate {total_words} words in chunks...")
                
                # Split into chunks
                chunks = []
                for i in range(0, total_words, CHUNK_SIZE_WORDS):
                    chunk_words = text_words[i:i + CHUNK_SIZE_WORDS]
                    chunks.append(" ".join(chunk_words))
                
                total_chunks = len(chunks)
                logger.info(f"🔄 Translating {total_words} words in {total_chunks} chunks on {device_label}...")
                
                # Set source language for tokenizer once
                tokenizer.src_lang = src_mbart
                tgt_lang_token_id = tokenizer.lang_code_to_id[tgt_mbart]
                
                translated_chunks = []
                for i, chunk in enumerate(chunks, 1):
                    chunk_progress = 10 + int((i - 1) / total_chunks * 80)  # 10-90%
                    
                    if progress_callback:
                        progress_callback(
                            chunk_progress,
                            f"Translating chunk {i}/{total_chunks} ({len(chunk.split())} words)..."
                        )
                    
                    # Tokenize chunk
                    encoded = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True, padding=True)
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    
                    # Generate translation
                    with torch.no_grad():
                        generated_tokens = model.generate(
                            **encoded,
                            forced_bos_token_id=tgt_lang_token_id,
                            max_length=1024,
                            num_beams=5,
                            early_stopping=True,
                            do_sample=False
                        )
                    
                    # Decode chunk translation
                    chunk_translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                    translated_chunks.append(chunk_translation)
                    
                    logger.debug(f"   ✓ Chunk {i}/{total_chunks} translated")
                
                # Combine translated chunks
                if progress_callback:
                    progress_callback(95, "Combining translated chunks...")
                
                translation = " ".join(translated_chunks)
                
                if progress_callback:
                    progress_callback(100, "Translation completed")
                
                logger.info(f"✅ Translated {total_words} words in {total_chunks} chunks on {device_label} from {LANG_NAMES.get(source_lang, source_lang)} to {LANG_NAMES.get(target_lang, target_lang)} ({len(translation.split())} words)")
                return translation
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if progress_callback:
                progress_callback(0, f"Translation error: {str(e)}")
            return text
    
    def _chunk_text(self, text: str, chunk_size: Optional[int] = None) -> List[str]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Input text to chunk.
            chunk_size: Size of each chunk in words (default: self.default_chunk_size).
            
        Returns:
            List of text chunks.
        """
        chunk_size = chunk_size or self.default_chunk_size
        words = text.split()
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks
    
    def _chunk_text_with_positions(
        self, 
        text: str, 
        chunk_size: Optional[int] = None
    ) -> List[Tuple[str, int, int]]:
        """
        Split text into chunks with character positions.
        
        Args:
            text: Input text to chunk.
            chunk_size: Size of each chunk in words.
            
        Returns:
            List of tuples: (chunk_text, start_pos, end_pos)
        """
        chunk_size = chunk_size or self.default_chunk_size
        words = text.split()
        chunks_with_pos = []
        
        current_pos = 0
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Find actual position in original text
            start_pos = text.find(chunk_words[0], current_pos) if chunk_words else current_pos
            end_pos = start_pos + len(chunk_text)
            current_pos = end_pos
            
            chunks_with_pos.append((chunk_text, start_pos, end_pos))
        
        return chunks_with_pos
    
    def _generate_part_title(self, part_number: int, total_parts: int, chunk_text: str) -> str:
        """
        Generate a title for a part summary.
        
        Args:
            part_number: The part number (1-indexed).
            total_parts: Total number of parts.
            chunk_text: The text content of this part.
            
        Returns:
            A descriptive title for the part.
        """
        # Extract first meaningful sentence or phrase for context
        sentences = re.split(r'[.!?]\s+', chunk_text[:200])
        first_sentence = sentences[0].strip() if sentences else ""
        
        # Truncate if too long
        if len(first_sentence) > 50:
            first_sentence = first_sentence[:47] + "..."
        
        if total_parts == 1:
            return "Summary"
        
        return f"Part {part_number} of {total_parts}"
    
    def _summarize_single_part(
        self,
        chunk_text: str,
        part_number: int,
        total_parts: int,
        start_pos: int,
        end_pos: int,
        compression_ratio: float = 0.4,
        extract_keywords: bool = False
    ) -> PartSummary:
        """
        Summarize a single part/chunk of text in English only.
        Text must already be in English before calling this method.
        
        Args:
            chunk_text: The text to summarize (must be in English).
            part_number: Part number (1-indexed).
            total_parts: Total number of parts.
            start_pos: Start character position in original text.
            end_pos: End character position in original text.
            compression_ratio: Target compression ratio.
            extract_keywords: Whether to extract keywords for this part.
            
        Returns:
            PartSummary object with the summary data.
        """
        chunk_words = chunk_text.split()
        word_count = len(chunk_words)
        
        # Generate title
        title = self._generate_part_title(part_number, total_parts, chunk_text)
        
        # Calculate summary length
        min_length, max_length = self._calculate_summary_length(word_count, compression_ratio)
        
        try:
            logger.info(f"📄 Summarizing {title}: {word_count} words -> target {min_length}-{max_length} words (English)")
            
            result = self.summarizer(
                chunk_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=4,  # Beam search for better quality
                length_penalty=0.8,  # Slight penalty for shorter summaries
                early_stopping=True
            )
            
            summary_text = result[0]['summary_text'] if isinstance(result, list) else result['summary_text']
            logger.info(f"✅ {title} summarized: {len(summary_text.split())} words")
            
        except Exception as e:
            logger.error(f"❌ Error summarizing {title}: {e}")
            # Fallback: use first portion of chunk
            summary_text = " ".join(chunk_words[:min(100, word_count)]) + "..."
        
        # Extract keywords if requested
        keywords = []
        if extract_keywords and self.kw_model:
            try:
                keywords = self.extract_keywords(chunk_text, top_n=5)
            except Exception as e:
                logger.warning(f"Keyword extraction failed for {title}: {e}")
        
        return PartSummary(
            part_number=part_number,
            title=title,
            summary=summary_text,
            word_count=word_count,
            start_position=start_pos,
            end_position=end_pos,
            keywords=keywords
        )
    
    def _generate_overview(self, part_summaries: List[PartSummary], compression_ratio: float = 0.5) -> str:
        """
        Generate an overview summary from all part summaries.
        
        Args:
            part_summaries: List of PartSummary objects.
            compression_ratio: Compression ratio for the overview.
            
        Returns:
            Overview summary string.
        """
        if not part_summaries:
            return ""
        
        if len(part_summaries) == 1:
            return part_summaries[0].summary
        
        # Combine all part summaries
        combined_text = " ".join([p.summary for p in part_summaries])
        combined_words = combined_text.split()
        
        # If combined is short enough, return as is
        if len(combined_words) < 150:
            return combined_text
        
        # Generate overview from combined summaries
        try:
            min_length, max_length = self._calculate_summary_length(
                len(combined_words), 
                compression_ratio
            )
            
            logger.info(f"📊 Generating overview from {len(part_summaries)} parts ({len(combined_words)} words)")
            
            result = self.summarizer(
                combined_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=4,  # Beam search for better quality
                length_penalty=0.8,  # Slight penalty for shorter summaries
                early_stopping=True
            )
            
            overview = result[0]['summary_text'] if isinstance(result, list) else result['summary_text']
            logger.info(f"✅ Overview generated: {len(overview.split())} words")
            return overview
            
        except Exception as e:
            logger.error(f"❌ Error generating overview: {e}")
            # Fallback: combine first sentences of each part
            overview_parts = []
            for p in part_summaries:
                sentences = re.split(r'[.!?]\s+', p.summary)
                if sentences:
                    overview_parts.append(sentences[0].strip())
            return ". ".join(overview_parts) + "."
    
    def summarize_parts(
        self,
        text: str,
        language: str = "AUTO",
        chunk_size: Optional[int] = None,
        compression_ratio: float = 0.4,
        generate_overview: bool = True,
        extract_part_keywords: bool = False
    ) -> StructuredSummary:
        """
        Summarize text into structured parts with proper architecture:
        
        STEP 1: Detect source language of transcript
        STEP 2A: Translate transcript to English (if source != English)
        STEP 2B: Summarize each part in English only (using BART/PEGASUS)
        STEP 3: Translate all part summaries to output language (if output != English)
        
        Args:
            text: Input text to summarize.
            language: Target OUTPUT language for summary. Accepts:
                     - "AUTO" to output in same language as transcript
                     - Language names: "English", "Chinese", "Spanish", etc.
                     - 2-letter codes: "en", "zh", "es", etc.
                     - mBART codes: "en_XX", "zh_CN", etc.
            chunk_size: Size of each part in words (default: self.default_chunk_size).
            compression_ratio: Target compression ratio for summaries.
            generate_overview: Whether to generate an overall overview.
            extract_part_keywords: Whether to extract keywords for each part.
            
        Returns:
            StructuredSummary containing all part summaries and overview.
            
        Raises:
            ValueError: If summarizer is not loaded or text is empty.
        """
        logger.info("=" * 60)
        logger.info("📑 PART SUMMARIZATION - 3 STEP ARCHITECTURE")
        logger.info("=" * 60)
        
        if not self.summarizer:
            raise ValueError("Summarization model not loaded")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # ============================================================
        # STEP 1: DETECT SOURCE LANGUAGE
        # ============================================================
        cleaned_text = self._clean_transcript_text(text)
        original_word_count = len(cleaned_text.split())
        source_lang = self._detect_language(cleaned_text)
        
        logger.info(f"📥 STEP 1 - DETECT SOURCE LANGUAGE")
        logger.info(f"   Transcript: {original_word_count} words in {LANG_NAMES.get(source_lang, source_lang)}")
        
        # Determine user's desired OUTPUT language
        language_code = normalize_language_code(language)
        if language_code == "AUTO":
            output_lang = source_lang
            logger.info(f"   Output language: AUTO -> {LANG_NAMES.get(output_lang, output_lang)}")
        else:
            output_lang = MBART_TO_LANG.get(language_code, "en")
            logger.info(f"   Output language: {language} -> {LANG_NAMES.get(output_lang, output_lang)}")
        
        # Check if text is too short for parts
        if original_word_count < self.min_words_for_summary:
            logger.info(f"   Text too short ({original_word_count} words), returning as-is")
            
            summary_text = cleaned_text
            # Translate if needed
            if source_lang != output_lang:
                summary_text = self._translate(cleaned_text, source_lang, output_lang)
            
            single_part = PartSummary(
                part_number=1,
                title="Summary",
                summary=summary_text,
                word_count=original_word_count,
                start_position=0,
                end_position=len(cleaned_text),
                keywords=[]
            )
            
            if extract_part_keywords and self.kw_model:
                try:
                    single_part.keywords = self.extract_keywords(cleaned_text, top_n=5)
                except:
                    pass
            
            return StructuredSummary(
                parts=[single_part],
                overview=summary_text,
                total_parts=1,
                original_word_count=original_word_count,
                summary_word_count=original_word_count,
                source_language=LANG_NAMES.get(source_lang, source_lang),
                target_language=LANG_NAMES.get(output_lang, output_lang)
            )
        
        # ============================================================
        # STEP 2A: TRANSLATE TRANSCRIPT TO ENGLISH (if needed)
        # Summarization always happens in English using BART/PEGASUS
        # ============================================================
        text_for_summarization = cleaned_text
        
        if source_lang != "en":
            logger.info(f"")
            logger.info(f"🔄 STEP 2A - TRANSLATE TRANSCRIPT TO ENGLISH")
            logger.info(f"   Translating from {LANG_NAMES.get(source_lang, source_lang)} -> English")
            text_for_summarization = self._translate(cleaned_text, source_lang, "en")
            logger.info(f"   ✓ Transcript translated to English ({len(text_for_summarization.split())} words)")
        else:
            logger.info(f"")
            logger.info(f"📝 STEP 2A - TRANSCRIPT ALREADY IN ENGLISH (no translation needed)")
        
        # ============================================================
        # STEP 2B: SUMMARIZE EACH PART IN ENGLISH
        # ============================================================
        logger.info(f"")
        logger.info(f"📝 STEP 2B - SUMMARIZE IN ENGLISH (using BART/PEGASUS)")
        
        # Chunk the text for summarization
        chunks_with_pos = self._chunk_text_with_positions(text_for_summarization, chunk_size)
        total_parts = len(chunks_with_pos)
        logger.info(f"   Divided into {total_parts} parts")
        
        # Summarize each part in English
        part_summaries: List[PartSummary] = []
        
        for i, (chunk_text, start_pos, end_pos) in enumerate(chunks_with_pos, 1):
            part_summary = self._summarize_single_part(
                chunk_text=chunk_text,
                part_number=i,
                total_parts=total_parts,
                start_pos=start_pos,
                end_pos=end_pos,
                compression_ratio=compression_ratio,
                extract_keywords=extract_part_keywords
            )
            logger.info(f"   ✓ Part {i}/{total_parts}: {len(part_summary.summary.split())} words in English")
            part_summaries.append(part_summary)
        
        # Generate overview in English
        overview_in_english = ""
        if generate_overview:
            overview_in_english = self._generate_overview(part_summaries, compression_ratio=0.5)
            logger.info(f"   ✓ Overview: {len(overview_in_english.split())} words in English")
        
        # ============================================================
        # STEP 3: TRANSLATE SUMMARIES TO OUTPUT LANGUAGE (if needed)
        # ============================================================
        logger.info(f"")
        logger.info(f"🔄 STEP 3 - TRANSLATE TO OUTPUT LANGUAGE")
        
        # Translate summaries to output language if needed
        if output_lang != "en":
            logger.info(f"   Translating: English -> {LANG_NAMES.get(output_lang, output_lang)}")
            for i, part in enumerate(part_summaries, 1):
                part.summary = self._translate(part.summary, "en", output_lang)
                logger.info(f"   ✓ Part {i} translated")
            
            if overview_in_english:
                overview = self._translate(overview_in_english, "en", output_lang)
                logger.info(f"   ✓ Overview translated")
            else:
                overview = ""
        else:
            logger.info(f"   No translation needed (output language is English)")
            overview = overview_in_english
        
        # Calculate total summary words
        summary_word_count = sum(len(p.summary.split()) for p in part_summaries)
        
        logger.info("=" * 60)
        logger.info(f"✅ FINAL OUTPUT: {total_parts} parts, {summary_word_count} words in {LANG_NAMES.get(output_lang, output_lang)}")
        logger.info("=" * 60)
        
        return StructuredSummary(
            parts=part_summaries,
            overview=overview,
            total_parts=total_parts,
            original_word_count=original_word_count,
            summary_word_count=summary_word_count,
            source_language=LANG_NAMES.get(source_lang, source_lang),
            target_language=LANG_NAMES.get(output_lang, output_lang)
        )
    
    def _calculate_summary_length(self, input_length: int, compression_ratio: float = 0.4) -> Tuple[int, int]:
        """
        Calculate optimal min and max length for summary based on input length.
        
        Args:
            input_length: Number of words in input text.
            compression_ratio: Target compression ratio (default: 0.4 = 40% of original).
            
        Returns:
            Tuple of (min_length, max_length) for summarization.
        """
        # Improved calculation to generate more comprehensive summaries
        # For longer texts, allow summaries up to 300 tokens
        # Minimum length is based on a percentage but ensures at least 30 words for better quality
        calculated_max = int(input_length * compression_ratio)
        
        # Scale max_length based on input size:
        # - Small texts (< 100 words): use calculated value, cap at 100
        # - Medium texts (100-500 words): cap at 200
        # - Large texts (> 500 words): cap at 300 for comprehensive summaries
        if input_length < 100:
            max_length = max(30, min(calculated_max, 100))
        elif input_length < 500:
            max_length = max(50, min(calculated_max, 200))
        else:
            max_length = max(80, min(calculated_max, 300))
        
        # Minimum length: ensure at least 50% of max_length or 15% of input_length, whichever is smaller
        # But never less than 20 words for readability and quality
        min_length = max(20, min(int(max_length * 0.5), int(input_length * 0.15)))
        
        # Ensure min_length is always less than max_length
        min_length = min(min_length, max_length - 5) if max_length > 25 else min_length
        
        return min_length, max_length
    
    def summarize(
        self,
        text: str,
        language: str = "AUTO",
        chunk_size: Optional[int] = None,
        compression_ratio: float = 0.4,
        return_original_if_short: bool = True,
        return_structured: bool = False,
        include_part_headers: bool = False
    ) -> str:
        """
        Summarize text with proper 3-step architecture:
        
        STEP 1: Detect source language
        STEP 2A: Translate transcript to English (if source != English)
        STEP 2B: Summarize in English only (using BART/PEGASUS)
        STEP 3: Translate summary to output language (if output != English)
        
        Args:
            text: Input text to summarize.
            language: Target OUTPUT language for summary. Accepts:
                     - "AUTO" to output in same language as transcript
                     - Language names: "English", "Chinese", "Spanish", etc.
                     - 2-letter codes: "en", "zh", "es", etc.
                     - mBART codes: "en_XX", "zh_CN", etc.
            chunk_size: Size of chunks for processing (default: self.default_chunk_size).
            compression_ratio: Target compression ratio for summary (default: 0.4).
            return_original_if_short: If True, return original text if it's too short.
            return_structured: If True, uses the new part-based summarization internally.
            include_part_headers: If True and return_structured, include part headers in output.
            
        Returns:
            Summarized text in user's chosen output language.
            
        Raises:
            ValueError: If summarizer is not loaded or text is empty.
        """
        logger.info("=" * 60)
        logger.info("🔍 SUMMARIZATION - 3 STEP ARCHITECTURE")
        logger.info("=" * 60)
        
        if not self.summarizer:
            raise ValueError("Summarization model not loaded")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # ============================================================
        # STEP 1: DETECT SOURCE LANGUAGE
        # ============================================================
        cleaned_text = self._clean_transcript_text(text)
        source_lang = self._detect_language(cleaned_text)
        
        logger.info(f"📥 STEP 1 - DETECT SOURCE LANGUAGE")
        logger.info(f"   Transcript language: {LANG_NAMES.get(source_lang, source_lang)} ({source_lang})")
        logger.info(f"   Input text: {len(cleaned_text)} chars, {len(cleaned_text.split())} words")
        
        # Determine user's desired OUTPUT language
        language_code = normalize_language_code(language)
        if language_code == "AUTO":
            # AUTO means output in same language as transcript
            output_lang = source_lang
            logger.info(f"   Output language: AUTO -> {LANG_NAMES.get(output_lang, output_lang)} (same as source)")
        else:
            output_lang = MBART_TO_LANG.get(language_code, "en")
            logger.info(f"   Output language: {language} -> {LANG_NAMES.get(output_lang, output_lang)}")
        
        # Use structured approach if requested
        if return_structured:
            structured = self.summarize_parts(
                text=text,
                language=language,
                chunk_size=chunk_size,
                compression_ratio=compression_ratio,
                generate_overview=True
            )
            return structured.get_combined_summary(include_part_headers=include_part_headers)
        
        words = cleaned_text.split()
        total_words = len(words)
        
        # Check if text is too short for summarization
        if total_words < self.min_words_for_summary:
            if return_original_if_short:
                logger.info(f"   Text too short ({total_words} words), returning original")
                if source_lang != output_lang:
                    return self._translate(cleaned_text, source_lang, output_lang)
                return cleaned_text
            else:
                raise ValueError(f"Text too short for summarization (minimum {self.min_words_for_summary} words)")
        
        # ============================================================
        # STEP 2A: TRANSLATE TRANSCRIPT TO ENGLISH (if needed)
        # Summarization always happens in English using BART/PEGASUS
        # ============================================================
        text_for_summarization = cleaned_text
        
        if source_lang != "en":
            logger.info(f"")
            logger.info(f"🔄 STEP 2A - TRANSLATE TRANSCRIPT TO ENGLISH")
            logger.info(f"   Translating from {LANG_NAMES.get(source_lang, source_lang)} -> English")
            text_for_summarization = self._translate(cleaned_text, source_lang, "en")
            logger.info(f"   ✓ Transcript translated to English ({len(text_for_summarization.split())} words)")
        else:
            logger.info(f"")
            logger.info(f"📝 STEP 2A - TRANSCRIPT ALREADY IN ENGLISH (no translation needed)")
        
        # ============================================================
        # STEP 2B: SUMMARIZE IN ENGLISH
        # ============================================================
        logger.info(f"")
        logger.info(f"📝 STEP 2B - SUMMARIZE IN ENGLISH (using BART/PEGASUS)")
        
        chunk_size = chunk_size or self.default_chunk_size
        chunks = self._chunk_text(text_for_summarization, chunk_size)
        
        summaries = []
        
        for i, chunk in enumerate(chunks):
            try:
                chunk_words = chunk.split()
                input_len = len(chunk_words)
                
                min_length, max_length = self._calculate_summary_length(input_len, compression_ratio)
                
                logger.info(f"   Chunk {i+1}/{len(chunks)}: {input_len} words -> target {min_length}-{max_length}")
                
                # Use beam search for better quality summaries
                # num_beams=4 provides a good balance between quality and speed
                # length_penalty encourages summaries closer to target length
                result = self.summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    num_beams=4,  # Beam search for better quality
                    length_penalty=0.8,  # Slight penalty for shorter summaries to encourage more complete ones
                    early_stopping=True  # Stop when all beams reach EOS
                )
                
                summary_text = result[0]['summary_text'] if isinstance(result, list) else result['summary_text']
                logger.info(f"   ✓ Chunk {i+1} summarized: {len(summary_text.split())} words in English")
                summaries.append(summary_text)
                
            except Exception as e:
                logger.error(f"   ✗ Error summarizing chunk {i+1}: {e}")
                # Fallback: use first portion of chunk
                chunk_words = chunk.split()[:100]
                fallback_text = " ".join(chunk_words) + "..."
                summaries.append(fallback_text)
        
        # Combine chunk summaries
        summary_in_english = " ".join(summaries)
        logger.info(f"   Combined summary: {len(summary_in_english.split())} words in English")
        logger.info(f"   Preview: {summary_in_english[:150]}...")
        
        # ============================================================
        # STEP 3: TRANSLATE SUMMARY TO OUTPUT LANGUAGE (if needed)
        # ============================================================
        logger.info(f"")
        logger.info(f"🔄 STEP 3 - TRANSLATE TO OUTPUT LANGUAGE")
        
        # Translate summary to output language if needed
        if output_lang != "en":
            logger.info(f"   Translating: English -> {LANG_NAMES.get(output_lang, output_lang)}")
            final_summary = self._translate(summary_in_english, "en", output_lang)
            logger.info(f"   ✓ Translation complete: {len(final_summary.split())} words")
            logger.info(f"   Preview: {final_summary[:150]}...")
        else:
            logger.info(f"   No translation needed (output language is English)")
            final_summary = summary_in_english
        
        logger.info("=" * 60)
        logger.info(f"✅ LANGUAGE FLOW: {LANG_NAMES.get(source_lang, source_lang)} -> English -> {LANG_NAMES.get(output_lang, output_lang)}")
        logger.info(f"✅ FINAL OUTPUT: {len(final_summary.split())} words in {LANG_NAMES.get(output_lang, output_lang)}")
        logger.info("=" * 60)
        
        return final_summary
    
    def summarize_with_keywords(
        self,
        text: str,
        language: str = "AUTO",
        top_n_keywords: Optional[int] = None,
        chunk_size: Optional[int] = None,
        compression_ratio: float = 0.4,
        return_original_if_short: bool = True,
        return_structured: bool = False
    ) -> Dict[str, Any]:
        """
        Summarize text and extract keywords in one call.
        
        Args:
            text: Input text to process.
            language: Target language for summary output. Accepts:
                     - "AUTO" to auto-detect and use source language
                     - Language names: "English", "Chinese", "Spanish", etc.
                     - 2-letter codes: "en", "zh", "es", etc.
                     - mBART codes: "en_XX", "zh_CN", etc.
            top_n_keywords: Number of keywords to extract (default: self.default_keyword_count).
            chunk_size: Size of chunks for summarization.
            compression_ratio: Target compression ratio for summary.
            return_original_if_short: If True, return original text if it's too short.
            return_structured: If True, include structured part summaries in result.
            
        Returns:
            Dictionary with keys:
            - 'summary': Summarized text (combined from all parts)
            - 'keywords': List of extracted keywords
            - 'original_length': Original text word count
            - 'summary_length': Summary word count
            - 'compression_ratio': Ratio of summary to original
            - 'target_language': The language of the summary output
            - 'parts': (if return_structured=True) List of part summary dicts
            - 'overview': (if return_structured=True) Overall summary
        """
        if not self.is_ready():
            raise ValueError("Both summarizer and keyword models must be loaded")
        
        # Clean text for both keywords and summary (remove timestamps)
        cleaned_text = self._clean_transcript_text(text)
        
        # Extract keywords (works on cleaned text, timestamps removed)
        try:
            keywords = self.extract_keywords(cleaned_text, top_n=top_n_keywords)
        except Exception as e:
            logger.warning(f"Keyword extraction failed, continuing without keywords: {e}")
            keywords = []
        
        if return_structured:
            # Use structured part summarization
            structured = self.summarize_parts(
                text=text,
                language=language,
                chunk_size=chunk_size,
                compression_ratio=compression_ratio,
                generate_overview=True,
                extract_part_keywords=False  # Already extracting global keywords
            )
            
            # Get combined summary (without headers for backward compatibility)
            summary = structured.get_combined_summary(include_part_headers=False)
            summary = self._clean_transcript_text(summary)
            
            original_length = structured.original_word_count
            summary_length = structured.summary_word_count
            
            return {
                'summary': summary,
                'keywords': keywords,
                'original_length': original_length,
                'summary_length': summary_length,
                'compression_ratio': summary_length / original_length if original_length > 0 else 0,
                'parts': [p.to_dict() for p in structured.parts],
                'overview': structured.overview,
                'total_parts': structured.total_parts,
                'source_language': structured.source_language,
                'target_language': structured.target_language
            }
        
        # Normalize language and detect for response
        language_code = normalize_language_code(language)
        source_lang = self._detect_language(cleaned_text)
        if language_code == "AUTO":
            target_lang = source_lang
        else:
            target_lang = MBART_TO_LANG.get(language_code, "en")
        
        # Generate summary using standard method
        summary = self.summarize(
            text=text,
            language=language,
            chunk_size=chunk_size,
            compression_ratio=compression_ratio,
            return_original_if_short=return_original_if_short
        )
        
        # Clean summary output to ensure no timestamps leaked through
        summary = self._clean_transcript_text(summary)
        
        original_length = len(text.split())
        summary_length = len(summary.split())
        
        return {
            'summary': summary,
            'keywords': keywords,
            'original_length': original_length,
            'summary_length': summary_length,
            'compression_ratio': summary_length / original_length if original_length > 0 else 0,
            'source_language': LANG_NAMES.get(source_lang, source_lang),
            'target_language': LANG_NAMES.get(target_lang, target_lang)
        }


def create_summary_service(
    summarizer: Optional[pipeline] = None,
    kw_model: Optional[KeyBERT] = None
) -> SummaryService:
    """
    Factory function to create a SummaryService instance.
    
    Args:
        summarizer: Optional pre-loaded summarization pipeline.
        kw_model: Optional pre-loaded KeyBERT model.
        
    Returns:
        Configured SummaryService instance.
    """
    return SummaryService(summarizer=summarizer, kw_model=kw_model)
