# TubeScribe AI Test Suite

This directory contains comprehensive tests for the TubeScribe AI backend.

## Test Coverage

The test suite covers:

1. **Transcription Tests** (`TestTranscribeEndpoint`)
   - Successful transcription
   - Invalid URL handling
   - Missing model error handling
   - Download error handling
   - Timestamp formatting
   - Transcript format for translation

2. **Summarization Tests** (`TestSummarizeEndpoint`)
   - Successful summarization (English)
   - Chinese language support
   - Spanish language support
   - Empty text validation
   - Default language handling
   - Service readiness checks

3. **Translation Integration Tests** (`TestTranslationIntegration`)
   - Transcript format compatibility with frontend translation
   - Summary format compatibility with frontend translation

4. **Progress Tracking Tests** (`TestProgressTracking`)
   - Progress retrieval for unknown videos
   - Progress retrieval for existing videos

5. **Service Unit Tests** (`TestSummaryService`)
   - Service readiness checks
   - Model validation
   - Error handling

6. **Utility Tests** (`TestVideoIDExtraction`)
   - YouTube URL parsing
   - Various URL formats

## Prerequisites

Install test dependencies:

```bash
pip install -r requirements.txt
```

The requirements.txt includes:
- `pytest>=7.4.0` - Testing framework
- `pytest-asyncio>=0.21.0` - Async test support
- `httpx>=0.24.0` - HTTP client for testing

## Running Tests

### Run All Tests

```bash
pytest test_tubescribe.py -v
```

### Run Specific Test Class

```bash
pytest test_tubescribe.py::TestTranscribeEndpoint -v
```

### Run Specific Test

```bash
pytest test_tubescribe.py::TestTranscribeEndpoint::test_transcribe_success -v
```

### Run with Coverage

```bash
pytest test_tubescribe.py --cov=app --cov=summary --cov-report=html
```

### Run Only Fast Tests (Skip Slow/Integration)

```bash
pytest test_tubescribe.py -v -m "not slow"
```

## Test Structure

Tests are organized into classes:

- **TestVideoIDExtraction**: URL parsing utilities
- **TestHealthEndpoints**: Health check and root endpoints
- **TestTranscribeEndpoint**: Transcription functionality
- **TestSummarizeEndpoint**: Summarization functionality
- **TestProgressTracking**: Progress tracking endpoints
- **TestSummaryService**: SummaryService unit tests
- **TestTranslationIntegration**: Translation integration tests

## Mocking

Tests use mocking to avoid:
- Downloading actual YouTube videos
- Loading heavy AI models (when not available)
- Making external API calls

Key mocks:
- `download_audio()` - Mocked to avoid actual downloads
- `transcriber` - Mocked when models aren't loaded
- `summary_service` - Uses actual service when available

## Notes

### Model Loading

Some tests will skip if models aren't loaded:
- Transcription tests skip if `transcriber` is `None`
- Summarization tests skip if `summary_service` is not ready

To run full tests, ensure models are loaded by starting the server first:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then run tests in a separate terminal.

### Translation Tests

Translation is handled on the frontend via Gemini API, not in the Python backend. The translation tests verify:
- Transcript format is compatible with frontend translation
- Summary format is compatible with frontend translation
- Timestamps are preserved correctly

## Continuous Integration

To integrate with CI/CD:

```yaml
# Example GitHub Actions
- name: Install dependencies
  run: pip install -r requirements.txt

- name: Run tests
  run: pytest test_tubescribe.py -v
```

## Troubleshooting

### Tests fail with "Model not loaded"

This is expected if models aren't loaded. Tests will skip gracefully. To run full tests:
1. Start the server to load models
2. Run tests in separate terminal
3. Or mock the models properly

### Import errors

Ensure you're running tests from the project root:
```bash
cd /path/to/tubescribe-ai
pytest test_tubescribe.py
```

### Permission errors (Windows)

If you encounter permission errors:
```powershell
.\scripts\fix-permissions.ps1
```
