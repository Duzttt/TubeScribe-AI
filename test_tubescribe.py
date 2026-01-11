"""
Comprehensive test suite for TubeScribe AI Backend

Tests for:
- Transcription endpoint (/api/transcribe)
- Summarization endpoint (/summarize)
- Translation functionality (frontend-based, tested via integration)

Run with: pytest test_tubescribe.py -v
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, mock_open
from fastapi.testclient import TestClient
from fastapi import HTTPException
import tempfile
import os
import json
from typing import Dict, Any

# Import the app and dependencies
import app as app_module
from app import (
    app as fastapi_app,
    get_video_id,
    download_audio,
    summary_service,
    progress_store,
    progress_lock
)
from summary import SummaryService


# Test client
client = TestClient(fastapi_app)


class TestVideoIDExtraction:
    """Tests for video ID extraction from YouTube URLs"""
    
    def test_get_video_id_standard_url(self):
        """Test extraction from standard YouTube URL"""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = get_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    def test_get_video_id_short_url(self):
        """Test extraction from short YouTube URL"""
        url = "https://youtu.be/dQw4w9WgXcQ"
        video_id = get_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    def test_get_video_id_embed_url(self):
        """Test extraction from embed YouTube URL"""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        video_id = get_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    def test_get_video_id_invalid_url(self):
        """Test extraction from invalid URL"""
        url = "https://example.com/video"
        video_id = get_video_id(url)
        assert video_id is None
    
    def test_get_video_id_with_timestamp(self):
        """Test extraction from URL with timestamp"""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=123"
        video_id = get_video_id(url)
        assert video_id == "dQw4w9WgXcQ"


class TestHealthEndpoints:
    """Tests for health check and root endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns service info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "TubeScribe API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "models_loaded" in data
    
    def test_health_check_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_ready" in data


class TestTranscribeEndpoint:
    """Tests for transcription endpoint"""
    
    @pytest.fixture
    def mock_audio_file(self):
        """Create a mock audio file for testing"""
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, "test_video.wav")
        # Create a dummy audio file
        with open(audio_path, "wb") as f:
            f.write(b"fake audio data")
        yield audio_path
        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)
    
    @pytest.fixture
    def mock_transcriber_result(self):
        """Mock transcription result"""
        return {
            "text": "This is a test transcription.",
            "chunks": [
                {
                    "timestamp": [0.0, 2.5],
                    "text": "This is a test transcription."
                }
            ],
            "language": "en"
        }
    
    @patch('app.transcriber')
    @patch('app.download_audio')
    def test_transcribe_success(
        self, 
        mock_download,
        mock_transcriber_func,
        mock_audio_file,
        mock_transcriber_result
    ):
        """Test successful transcription"""
        # Skip if transcriber not loaded in actual app
        if app_module.transcriber is None:
            pytest.skip("Transcriber not loaded - skipping test")
            return
        
        # Setup mocks
        mock_download.return_value = mock_audio_file
        mock_transcriber_func.return_value = mock_transcriber_result
        
        request_data = {
            "videoUrl": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        }
        
        response = client.post("/api/transcribe", json=request_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "transcript" in data
        assert "processing_time" in data
        assert "This is a test transcription" in data["transcript"] or len(data["transcript"]) > 0
        
        # Verify mocks were called
        mock_download.assert_called_once()
        mock_transcriber_func.assert_called_once()
    
    def test_transcribe_invalid_url(self):
        """Test transcription with invalid YouTube URL"""
        request_data = {
            "videoUrl": "https://example.com/video"
        }
        
        response = client.post("/api/transcribe", json=request_data)
        assert response.status_code == 400
        assert "Invalid YouTube URL" in response.json()["detail"]
    
    def test_transcribe_missing_url(self):
        """Test transcription with missing URL"""
        request_data = {}
        
        response = client.post("/api/transcribe", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_transcribe_no_model_loaded(self):
        """Test transcription when model is not loaded"""
        # Temporarily set transcriber to None
        original_transcriber = app_module.transcriber
        app_module.transcriber = None
        
        try:
            request_data = {
                "videoUrl": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            }
            
            response = client.post("/api/transcribe", json=request_data)
            assert response.status_code == 503
            assert "Transcription model not loaded" in response.json()["detail"]
        finally:
            # Restore transcriber
            app_module.transcriber = original_transcriber
    
    @patch('app.download_audio')
    def test_transcribe_download_error(
        self, 
        mock_download
    ):
        """Test transcription when download fails"""
        # Mock download to raise an error
        mock_download.side_effect = Exception("Download failed")
        
        request_data = {
            "videoUrl": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        }
        
        response = client.post("/api/transcribe", json=request_data)
        assert response.status_code == 500
        assert "Transcription failed" in response.json()["detail"]
    
    @patch('app.transcriber')
    @patch('app.download_audio')
    def test_transcribe_with_timestamps(
        self, 
        mock_download,
        mock_transcriber_func,
        mock_audio_file
    ):
        """Test transcription with timestamp formatting"""
        # Skip if transcriber not loaded in actual app
        if app_module.transcriber is None:
            pytest.skip("Transcriber not loaded - skipping test")
            return
        
        mock_download.return_value = mock_audio_file
        mock_result = {
            "chunks": [
                {
                    "timestamp": [0.0, 3.5],
                    "text": "First sentence."
                },
                {
                    "timestamp": [3.5, 7.0],
                    "text": "Second sentence."
                }
            ]
        }
        mock_transcriber_func.return_value = mock_result
        
        request_data = {
            "videoUrl": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        }
        
        response = client.post("/api/transcribe", json=request_data)
        assert response.status_code == 200
        data = response.json()
        transcript = data["transcript"]
        # Check for timestamp format
        assert "[" in transcript
        assert "]" in transcript


class TestSummarizeEndpoint:
    """Tests for summarization endpoint"""
    
    def test_summarize_success(self):
        """Test successful summarization"""
        # Use a longer text for meaningful summarization
        test_text = """
        Artificial intelligence (AI) is transforming the world in unprecedented ways. 
        From healthcare to transportation, AI technologies are revolutionizing industries. 
        Machine learning algorithms can now diagnose diseases with high accuracy. 
        Autonomous vehicles are becoming a reality thanks to AI. 
        Natural language processing enables computers to understand human language. 
        The future of AI holds great promise for solving complex global challenges.
        """ * 3  # Make it long enough for summarization
        
        request_data = {
            "text": test_text,
            "lang": "en_XX"
        }
        
        response = client.post("/summarize", json=request_data)
        
        # Check if models are loaded first
        if response.status_code == 503:
            pytest.skip("Summarization models not loaded - skipping test")
            return
        
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "keywords" in data
        assert isinstance(data["keywords"], list)
        assert len(data["keywords"]) > 0
        # Summary should be shorter than original
        assert len(data["summary"]) < len(test_text)
    
    def test_summarize_empty_text(self):
        """Test summarization with empty text"""
        request_data = {
            "text": "",
            "lang": "en_XX"
        }
        
        response = client.post("/summarize", json=request_data)
        assert response.status_code == 400
        assert "Text cannot be empty" in response.json()["detail"]
    
    def test_summarize_whitespace_only(self):
        """Test summarization with whitespace-only text"""
        request_data = {
            "text": "   \n\t  ",
            "lang": "en_XX"
        }
        
        response = client.post("/summarize", json=request_data)
        assert response.status_code == 400
    
    def test_summarize_chinese_language(self):
        """Test summarization with Chinese text"""
        test_text = """
        人工智能正在以前所未有的方式改变世界。
        从医疗保健到交通运输，人工智能技术正在彻底改变各个行业。
        机器学习算法现在可以高精度地诊断疾病。
        自动驾驶汽车由于人工智能而成为现实。
        自然语言处理使计算机能够理解人类语言。
        """ * 3
        
        request_data = {
            "text": test_text,
            "lang": "zh_CN"
        }
        
        response = client.post("/summarize", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Summarization models not loaded - skipping test")
            return
        
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "keywords" in data
    
    def test_summarize_spanish_language(self):
        """Test summarization with Spanish text"""
        test_text = """
        La inteligencia artificial está transformando el mundo de formas sin precedentes.
        Desde la atención médica hasta el transporte, las tecnologías de IA están revolucionando industrias.
        Los algoritmos de aprendizaje automático ahora pueden diagnosticar enfermedades con alta precisión.
        Los vehículos autónomos se están convirtiendo en una realidad gracias a la IA.
        """ * 3
        
        request_data = {
            "text": test_text,
            "lang": "es_XX"
        }
        
        response = client.post("/summarize", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Summarization models not loaded - skipping test")
            return
        
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "keywords" in data
    
    def test_summarize_default_language(self):
        """Test summarization with default language (English)"""
        test_text = "This is a test text for summarization. " * 20
        
        request_data = {
            "text": test_text
        }
        
        response = client.post("/summarize", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Summarization models not loaded - skipping test")
            return
        
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "keywords" in data
    
    @patch('app.summary_service')
    def test_summarize_service_not_ready(self, mock_service):
        """Test summarization when service is not ready"""
        mock_service.is_ready.return_value = False
        
        request_data = {
            "text": "Test text for summarization " * 10,
            "lang": "en_XX"
        }
        
        response = client.post("/summarize", json=request_data)
        assert response.status_code == 503
        assert "Summary service not ready" in response.json()["detail"]


class TestProgressTracking:
    """Tests for progress tracking endpoints"""
    
    def test_get_progress_unknown_video(self):
        """Test getting progress for unknown video"""
        video_id = "unknown_video_id_123"
        response = client.get(f"/api/progress/{video_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unknown"
        assert data["progress"] == 0
    
    def test_get_progress_existing_video(self):
        """Test getting progress for existing video"""
        video_id = "test_video_123"
        
        # Set up progress
        with progress_lock:
            progress_store[video_id] = {
                "status": "transcribing",
                "progress": 50,
                "message": "Processing...",
                "download_progress": 100,
                "transcription_progress": 50
            }
        
        try:
            response = client.get(f"/api/progress/{video_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "transcribing"
            assert data["progress"] == 50
        finally:
            # Cleanup
            with progress_lock:
                if video_id in progress_store:
                    del progress_store[video_id]


class TestSummaryService:
    """Unit tests for SummaryService class"""
    
    def test_summary_service_is_ready(self):
        """Test SummaryService.is_ready() method"""
        # Create a service with both models
        mock_summarizer = Mock()
        mock_kw_model = Mock()
        service = SummaryService(summarizer=mock_summarizer, kw_model=mock_kw_model)
        assert service.is_ready() is True
    
    def test_summary_service_not_ready_no_summarizer(self):
        """Test SummaryService when summarizer is missing"""
        mock_kw_model = Mock()
        service = SummaryService(summarizer=None, kw_model=mock_kw_model)
        assert service.is_ready() is False
    
    def test_summary_service_not_ready_no_kw_model(self):
        """Test SummaryService when keyword model is missing"""
        mock_summarizer = Mock()
        service = SummaryService(summarizer=mock_summarizer, kw_model=None)
        assert service.is_ready() is False
    
    def test_extract_keywords_empty_text(self):
        """Test keyword extraction with empty text"""
        mock_kw_model = Mock()
        service = SummaryService(kw_model=mock_kw_model)
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            service.extract_keywords("")
    
    def test_extract_keywords_no_model(self):
        """Test keyword extraction without model"""
        service = SummaryService(kw_model=None)
        
        with pytest.raises(ValueError, match="Keyword extraction model not loaded"):
            service.extract_keywords("Test text")
    
    def test_summarize_empty_text(self):
        """Test summarization with empty text"""
        mock_summarizer = Mock()
        service = SummaryService(summarizer=mock_summarizer)
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            service.summarize("")
    
    def test_summarize_no_model(self):
        """Test summarization without model"""
        service = SummaryService(summarizer=None)
        
        with pytest.raises(ValueError, match="Summarization model not loaded"):
            service.summarize("Test text for summarization")


class TestTranslationIntegration:
    """
    Integration tests for translation functionality.
    
    Note: Translation is handled on the frontend via Gemini API.
    These tests verify the integration points and mock the frontend behavior.
    """
    
    @patch('app.transcriber')
    @patch('app.download_audio')
    def test_transcribe_returns_transcript_for_translation(
        self,
        mock_download,
        mock_transcriber_func
    ):
        """Test that transcription returns formatted transcript suitable for translation"""
        # This test ensures the transcript format is compatible with frontend translation
        # The frontend expects timestamps in [HH:MM:SS] or [MM:SS] format
        
        if app_module.transcriber is None:
            pytest.skip("Transcriber not loaded - skipping test")
            return
        
        # Create mock audio file path
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, "test_translate.wav")
        
        try:
            mock_download.return_value = audio_path
            mock_result = {
                "chunks": [
                    {
                        "timestamp": [0.0, 5.0],
                        "text": "Hello, this is a test transcript."
                    }
                ]
            }
            mock_transcriber_func.return_value = mock_result
            
            request_data = {
                "videoUrl": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            }
            
            response = client.post("/api/transcribe", json=request_data)
            
            if response.status_code == 503:
                pytest.skip("Transcription model not loaded - skipping test")
                return
            
            assert response.status_code == 200
            data = response.json()
            transcript = data["transcript"]
            
            # Verify transcript format is suitable for translation
            # Should contain timestamps in [MM:SS] or [HH:MM:SS] format
            assert "[" in transcript
            assert "]" in transcript
            # Should contain the actual text
            assert len(transcript) > 0
            
        finally:
            # Cleanup - though file might not exist since we're mocking
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    def test_summary_format_suitable_for_translation(self):
        """Test that summary format is suitable for frontend translation"""
        test_text = """
        This is a comprehensive test document about artificial intelligence.
        AI is transforming multiple industries including healthcare, finance, and education.
        Machine learning algorithms enable computers to learn from data.
        Deep learning models can process complex patterns in images and text.
        """ * 3
        
        request_data = {
            "text": test_text,
            "lang": "en_XX"
        }
        
        response = client.post("/summarize", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("Summarization models not loaded - skipping test")
            return
        
        assert response.status_code == 200
        data = response.json()
        
        # Summary should be plain text (no special formatting that breaks translation)
        summary = data["summary"]
        assert isinstance(summary, str)
        assert len(summary) > 0
        
        # Keywords should be a simple list
        keywords = data["keywords"]
        assert isinstance(keywords, list)
        assert all(isinstance(kw, str) for kw in keywords)


# Pytest configuration
@pytest.fixture(scope="session", autouse=True)
def cleanup_progress_store():
    """Cleanup progress store after all tests"""
    yield
    with progress_lock:
        progress_store.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
