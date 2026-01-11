# TubeScribe AI - System Architecture

## Overview

TubeScribe AI is a modern full-stack application that provides AI-powered video transcription, summarization, translation, and interactive chat capabilities. The system is designed with a clear separation between frontend and backend, leveraging local AI models for core processing and optional cloud APIs for enhanced features.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  Frontend (React + TypeScript)             │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐ │  │
│  │  │ Components  │  │   Services   │  │   State Mgmt     │ │  │
│  │  │ - YouTube   │  │ - Gemini API │  │ - React Hooks    │ │  │
│  │  │ - ResultCard│  │ - Summarize  │  │ - Local Storage  │ │  │
│  │  │ - Chat      │  │ - Backend    │  │                  │ │  │
│  │  │ - Progress  │  │              │  │                  │ │  │
│  │  └─────────────┘  └──────────────┘  └──────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────────┘
                        │ HTTP/REST API
                        │ Server-Sent Events (SSE)
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              Python Backend (FastAPI + Uvicorn)                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  API Layer (app.py)                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │  │
│  │  │ Transcription│  │ Summarization│  │   Translation   │ │  │
│  │  │ Endpoint     │  │ Endpoint     │  │   Endpoint      │ │  │
│  │  │ /api/        │  │ /summarize   │  │  /api/translate │ │  │
│  │  │ transcribe   │  │              │  │                 │ │  │
│  │  └──────────────┘  └──────────────┘  └─────────────────┘ │  │
│  │  ┌───────────────────────────────────────────────────────┐ │  │
│  │  │           Progress Tracking (SSE)                     │ │  │
│  │  │  /api/progress/{video_id}/stream                      │ │  │
│  │  └───────────────────────────────────────────────────────┘ │  │
│  └─────────────────────────┬───────────────────────────────────┘  │
│                            │                                       │
│  ┌─────────────────────────▼───────────────────────────────────┐  │
│  │              AI Models & Processing Layer                   │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │  │
│  │  │   Whisper    │  │   mBART/BART │  │    KeyBERT      │   │  │
│  │  │ (Transcribe) │  │ (Summarize)  │  │  (Keywords)     │   │  │
│  │  │              │  │   mBART-50   │  │                 │   │  │
│  │  │              │  │ (Translate)  │  │                 │   │  │
│  │  └──────────────┘  └──────────────┘  └─────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────┘  │
└───────────────────────┬───────────────────────────────────────────┘
                        │
                        │ yt-dlp + FFmpeg
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Services                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐     │
│  │   YouTube    │  │ Gemini API   │  │   HuggingFace    │     │
│  │  (Video/Audio│  │  (Optional)  │  │   Model Hub      │     │
│  │   Download)  │  │ - Chat       │  │  (Model Downloads)│     │
│  │              │  │ - Translate  │  │                  │     │
│  └──────────────┘  └──────────────┘  └──────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## System Components

### 1. Frontend Layer

**Technology Stack:**
- **Framework**: React 18+ with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS with dark mode support
- **State Management**: React Hooks (useState, useEffect, useCallback)
- **HTTP Client**: Fetch API (native)
- **Real-time Updates**: EventSource API (Server-Sent Events)

**Key Components:**

#### 1.1 Core Application (`App.tsx`)
- Main application container and state management
- Orchestrates the processing workflow (Transcribe → Translate → Summarize)
- Manages theme (light/dark mode)
- Coordinates between components and services

#### 1.2 Input Components
- **`YouTubeInput.tsx`**: URL input and validation
- **`FileUpload.tsx`**: File upload interface (if supported)

#### 1.3 Display Components
- **`ResultCard.tsx`**: Displays transcripts, summaries, translations with markdown rendering
- **`KeywordsDisplay.tsx`**: Visual keyword/tag display
- **`ProgressBar.tsx`**: Real-time progress visualization
- **`ChatInterface.tsx`**: Interactive Q&A interface

#### 1.4 Service Layer
- **`summarizationService.ts`**: 
  - Communicates with Python backend for summarization and keyword extraction
  - Health check monitoring
  - Language code mapping (mBART format)
  
- **`geminiService.ts`**:
  - Transcription request routing to Python backend
  - Translation using Gemini API (optional fallback)
  - Chat functionality with context management

**Frontend State Management:**
```typescript
State Variables:
- videoUrl: Current YouTube URL
- transcript: Original transcript text
- translatedTranscript: Translated version
- summary: Generated summary
- keywords: Extracted keywords array
- status: ProcessingStatus enum
- targetLang: Selected target language
- chatMessages: Chat conversation history
- transcriptionProgress: Real-time progress data
```

### 2. Backend Layer (Python FastAPI)

**Technology Stack:**
- **Framework**: FastAPI with Uvicorn ASGI server
- **AI Models**: HuggingFace Transformers
- **Transcription**: OpenAI Whisper (via transformers pipeline)
- **Summarization**: Facebook mBART-50 or BART-Large-CNN
- **Translation**: mBART-50 (multilingual)
- **Keyword Extraction**: KeyBERT with multilingual embeddings
- **Video Processing**: yt-dlp + FFmpeg
- **Async Support**: asyncio for concurrent operations

**Key Modules:**

#### 2.1 API Server (`app.py`)
**Responsibilities:**
- HTTP endpoint definitions
- Request/response models (Pydantic)
- CORS middleware configuration
- Model initialization and lifecycle management
- Progress tracking and SSE streaming

**Key Endpoints:**
```
GET  /                          - Service info and model status
GET  /health                    - Health check
GET  /api/languages             - Supported languages list
POST /api/transcribe            - Transcribe YouTube video
GET  /api/progress/{video_id}   - Polling progress endpoint
GET  /api/progress/{video_id}/stream - SSE progress stream
POST /summarize                 - Summarize text + extract keywords
POST /api/translate             - Translate text (local mBART)
```

#### 2.2 Summary Service (`summary.py`)
**Responsibilities:**
- Text summarization using transformer models
- Multilingual support (50+ languages via mBART)
- Keyword extraction using KeyBERT
- Language detection and normalization
- Translation between languages

**Key Classes:**
- `SummaryService`: Main service class coordinating all summary operations
- `PartSummary`: Data class for chunked summaries
- `StructuredSummary`: Complete summary with parts and overview

#### 2.3 Transcription Pipeline
**Process Flow:**
1. Extract video ID from YouTube URL
2. Download audio using yt-dlp (with multiple fallback strategies)
3. Convert audio to WAV format using FFmpeg
4. Process audio chunks with Whisper model
5. Format transcript with timestamps
6. Clean up temporary files

**Progress Tracking:**
- Real-time progress updates via SSE
- Two-phase progress: Download (0-30%) and Transcription (30-100%)
- Queue-based event distribution to multiple clients

### 3. AI Models Architecture

#### 3.1 Transcription Model (Whisper)
- **Model**: `openai/whisper-base`
- **Type**: Automatic Speech Recognition (ASR)
- **Capabilities**:
  - Multilingual speech recognition
  - Automatic language detection
  - Timestamp generation
  - Handles various accents and audio qualities

- **Device**: GPU (CUDA) if available, falls back to CPU
- **Processing**: Chunked processing (30-second chunks) for long videos

#### 3.2 Summarization Models
**Primary Model**: `facebook/mbart-large-50-many-to-many-mmt`
- **Type**: Sequence-to-sequence transformer
- **Capabilities**:
  - Multilingual summarization (50+ languages)
  - Abstractive summaries (generates new text)
  - Language code mapping support

**Fallback Model**: `facebook/bart-large-cnn`
- **Type**: English-only summarization
- **Use Case**: Faster processing for English content

**Model Selection Logic:**
- Environment variable: `SUMMARIZATION_MODEL`
- Automatic detection of multilingual vs. English-only models
- Fallback mechanism if primary model fails to load

#### 3.3 Translation Model
- **Model**: mBART-50 (embedded in SummaryService)
- **Capabilities**: Translation between 50+ language pairs
- **Usage**: Local translation without external API calls

#### 3.4 Keyword Extraction
- **Model**: KeyBERT with `paraphrase-multilingual-MiniLM-L12-v2`
- **Method**: Embedding-based keyword extraction
- **Output**: Top N keywords with relevance scores

### 4. Data Flow

#### 4.1 Transcription Flow
```
User Input (YouTube URL)
    ↓
Frontend: handleGenerateTranscript()
    ↓
Establish SSE Connection: /api/progress/{video_id}/stream
    ↓
POST /api/transcribe
    ↓
Backend: Extract video ID
    ↓
Backend: download_audio() with progress tracking
    ├─→ Update progress (0-30%)
    └─→ yt-dlp + FFmpeg processing
    ↓
Backend: Transcribe with Whisper
    ├─→ Update progress (30-100%)
    └─→ Chunk processing with timestamps
    ↓
Format transcript with timestamps
    ↓
Return TranscribeResponse
    ↓
Frontend: Display transcript in ResultCard
```

#### 4.2 Summarization Flow
```
User clicks "Summarize"
    ↓
Frontend: Check backend health
    ↓
If translation needed:
    ├─→ Translate transcript (Gemini API or local mBART)
    └─→ Cache translated version
    ↓
POST /summarize with text + language code
    ↓
Backend: SummaryService.summarize_with_keywords()
    ├─→ Language detection/normalization
    ├─→ Text chunking (if long)
    ├─→ Summarize each chunk
    ├─→ Extract keywords
    └─→ Combine results
    ↓
Return SummarizeResponse {summary, keywords}
    ↓
Frontend: Display summary + keywords
```

#### 4.3 Progress Tracking (SSE)
```
Backend: Transcription starts
    ↓
Create progress_store entry for video_id
    ↓
Client: Connect to /api/progress/{video_id}/stream
    ↓
Backend: Create event queue for client
    ↓
During processing:
    ├─→ Update progress_store
    └─→ Put update in queue
    ↓
SSE Generator: Poll queue every 50ms
    ├─→ Send updates to client
    └─→ Keep-alive messages every 5s
    ↓
Client: Receive events via EventSource.onmessage
    ↓
Frontend: Update UI with smooth animations
```

### 5. Communication Protocols

#### 5.1 REST API
- **Protocol**: HTTP/1.1 over TCP
- **Content Type**: JSON (application/json)
- **CORS**: Enabled for all origins (development)
- **Timeout**: 10 minutes for summarization (long operations)

#### 5.2 Server-Sent Events (SSE)
- **Protocol**: HTTP/1.1 with text/event-stream content type
- **Purpose**: Real-time progress updates
- **Format**: JSON payloads wrapped in SSE format
- **Keep-alive**: 5-second interval
- **Connection Management**: Automatic cleanup on completion/error

#### 5.3 Error Handling
- **HTTP Status Codes**: 
  - 200: Success
  - 400: Bad Request (invalid input)
  - 500: Internal Server Error
  - 503: Service Unavailable (models not loaded)
- **Error Response Format**: `{"detail": "error message"}`

### 6. Resource Management

#### 6.1 Memory Management
- **Model Loading**: Lazy initialization at startup
- **GPU Memory**: Automatic detection and allocation
- **Temporary Files**: Automatic cleanup after processing
- **Cache Management**: Transcript/summary caching in frontend state

#### 6.2 Processing Resources
- **CPU**: Primary processing (if GPU unavailable)
- **GPU**: Accelerated inference (CUDA 12.1+)
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: ~2-3GB for model files (first download)

#### 6.3 Concurrency
- **Async Operations**: FastAPI async endpoints
- **Background Tasks**: Audio cleanup after transcription
- **Multiple Clients**: Queue-based SSE for concurrent connections

### 7. Configuration & Environment

#### 7.1 Frontend Configuration
```env
# .env.local
VITE_API_KEY=your_gemini_api_key  # Optional
SUMMARIZATION_BACKEND_URL=http://localhost:8000
```

#### 7.2 Backend Configuration
```env
# Environment Variables
PORT=8000                           # Server port
HOST=0.0.0.0                        # Server host
SUMMARIZATION_MODEL=facebook/mbart-large-50-many-to-many-mmt
TRANSLATION_MODEL=null              # Optional: custom translation model
```

### 8. Security Considerations

#### 8.1 API Security
- **CORS**: Currently permissive (development mode)
- **Input Validation**: Pydantic models for request validation
- **URL Validation**: Regex-based YouTube URL validation
- **Error Messages**: Sanitized to prevent information leakage

#### 8.2 Model Security
- **Local Processing**: No data sent to external AI services (transcription/summarization)
- **API Keys**: Only used for optional features (translation/chat via Gemini)
- **Temporary Files**: Stored in system temp directory, auto-deleted

### 9. Scalability Considerations

#### 9.1 Current Limitations
- Single-threaded model inference
- Sequential request processing
- In-memory progress tracking (not persistent)
- **Language Support Limitation**: The summarization and translation models do not support Malay language. While Malay may appear in language lists due to configuration mappings, the underlying mBART and other models do not provide reliable support for Malay transcription, summarization, or translation. Users attempting to process Malay content may experience reduced accuracy or errors.

#### 9.2 Potential Improvements
- **Horizontal Scaling**: Multiple backend instances behind load balancer
- **Task Queue**: Redis/RabbitMQ for distributed processing
- **Model Caching**: Shared model storage (Redis/gRPC server)
- **Database**: Persistent storage for transcripts/summaries
- **CDN**: Static asset distribution for frontend

### 10. Deployment Architecture

#### 10.1 Development
```
Frontend:  Vite dev server (localhost:5173)
Backend:   Uvicorn with reload (localhost:8000)
Models:    Local download (~/.cache/huggingface)
```

#### 10.2 Production (Recommended)
```
Frontend:  Static build (nginx/cloud storage)
Backend:   Docker container with GPU support
           - Multi-stage build
           - Model pre-loading
           - Health checks
Models:    Persistent volume or model cache
```

## Technology Stack Summary

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend Framework | React 18 + TypeScript | UI development |
| Build Tool | Vite | Fast development & building |
| Styling | Tailwind CSS | Utility-first styling |
| Backend Framework | FastAPI | High-performance API server |
| ASGI Server | Uvicorn | ASGI application server |
| Transcription | OpenAI Whisper | Speech-to-text |
| Summarization | mBART-50 / BART | Text summarization |
| Translation | mBART-50 | Multilingual translation |
| Keywords | KeyBERT | Keyword extraction |
| Video Download | yt-dlp | YouTube video/audio extraction |
| Audio Processing | FFmpeg | Audio format conversion |
| Deep Learning | PyTorch | Model runtime |
| API Client | Fetch API / EventSource | HTTP communication |
| State Management | React Hooks | Frontend state |
| Validation | Pydantic | Request/response validation |

## Performance Characteristics

### Transcription
- **CPU**: ~1-2x real-time (e.g., 10min video = 10-20min processing)
- **GPU**: ~5-10x real-time (e.g., 10min video = 1-2min processing)
- **Model Size**: ~150MB (Whisper-base)
- **Memory**: ~1-2GB during processing

### Summarization
- **Speed**: ~100-500 words/second (GPU), ~20-100 words/second (CPU)
- **Model Size**: ~1.2GB (mBART-50), ~560MB (BART-Large)
- **Memory**: ~2-4GB during processing
- **Batch Processing**: Automatic chunking for long texts

### Translation
- **Speed**: ~50-200 words/second (GPU), ~10-50 words/second (CPU)
- **Model Size**: ~1.2GB (mBART-50)
- **Memory**: ~2-3GB during processing

## Monitoring & Observability

### Logging
- **Level**: INFO (production), DEBUG (development)
- **Format**: Structured logging with timestamps
- **Areas**: API requests, model operations, errors

### Health Checks
- **Endpoint**: `/health`
- **Checks**: Model availability, service status
- **Response Time**: <100ms

### Metrics (Potential)
- Request count
- Processing time
- Error rates
- Model inference time
- GPU utilization

## Future Architecture Enhancements

1. **Microservices**: Split transcription, summarization, and translation into separate services
2. **Message Queue**: Implement task queue for async processing
3. **Caching Layer**: Redis for transcript/summary caching
4. **Database**: PostgreSQL for persistent storage
5. **Authentication**: User accounts and API key management
6. **Rate Limiting**: Prevent abuse and manage resources
7. **Model Serving**: Dedicated model server (e.g., TorchServe, TensorFlow Serving)
8. **CDN Integration**: Fast global content delivery

---

**Last Updated**: Based on current codebase analysis
**Architecture Version**: 1.0
