# Summarization Model Options

The summarization model can be changed via the `SUMMARIZATION_MODEL` environment variable.

## Available Models

### 1. **facebook/mbart-large-50-many-to-many-mmt** (Default)
- **Languages**: 50+ languages including English, Chinese, Spanish, French, German, etc.
- **Best for**: Multilingual content
- **Size**: ~2GB
- **Speed**: Medium
- **Use case**: When you need to summarize content in multiple languages
- **⚠️ Limitation**: This model does **not support Malay language**. While Malay may appear in language configuration lists, the model does not provide reliable summarization or translation for Malay content.

### 2. **facebook/bart-large-cnn**
- **Languages**: English only
- **Best for**: English content, faster processing
- **Size**: ~1.6GB
- **Speed**: Fast
- **Use case**: English-only videos, better performance

### 3. **google/pegasus-xsum**
- **Languages**: English only
- **Best for**: Abstractive, concise summaries
- **Size**: ~0.57GB
- **Speed**: Fast
- **Use case**: When you want very concise, abstract summaries

### 4. **t5-base**
- **Languages**: Multilingual (requires proper tokenizer setup)
- **Best for**: General purpose, balanced performance
- **Size**: ~850MB
- **Speed**: Medium-Fast
- **Use case**: Balanced multilingual support with smaller size

### 5. **t5-large**
- **Languages**: Multilingual (requires proper tokenizer setup)
- **Best for**: Better quality than t5-base
- **Size**: ~3GB
- **Speed**: Medium
- **Use case**: Better quality with multilingual support

## How to Change the Model

### Windows (PowerShell)
```powershell
$env:SUMMARIZATION_MODEL="facebook/bart-large-cnn"
.\scripts\start-python-backend.ps1
```

### Windows (Command Prompt)
```cmd
set SUMMARIZATION_MODEL=facebook/bart-large-cnn
scripts\start-python-backend.bat
```

### Linux/Mac
```bash
export SUMMARIZATION_MODEL="facebook/bart-large-cnn"
./scripts/start-python-backend.sh
```

### Or modify directly in the code
Edit `app.py` line 70 and change:
```python
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "facebook/mbart-large-50-many-to-many-mmt")
```
to:
```python
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "facebook/bart-large-cnn")  # or your preferred model
```

## Notes

- First-time model loading will download the model (may take several minutes)
- Model size affects RAM usage (1-3GB typical)
- English-only models are faster but won't work for non-English content
- After changing the model, restart the backend server
- The model will be cached in your HuggingFace cache directory

## Known Limitations

### Malay Language Support
**The models used in this project do not support Malay language.** While Malay (language code: `ms`) may appear in some language lists or configuration files, the underlying AI models (including mBART-50, BART, and Whisper) do not provide reliable support for:

- Malay transcription accuracy
- Malay summarization quality
- Malay translation functionality

Users attempting to process Malay content may experience:
- Reduced transcription accuracy
- Poor or incorrect summarization results
- Translation errors or failures
- Unexpected behavior in language detection

**Recommendation**: For Malay content, consider using alternative tools or models specifically trained on Malay language data.
