"""
Summary Module for TubeScribe AI

Provides multilingual text summarization and keyword extraction functionality.
Uses mBART for summarization and KeyBERT for keyword extraction.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
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
    "nl": "nl_XX", "tr": "tr_TR", "vi": "vi_VN"
}

# Reverse mapping (mBART code -> 2-letter code)
MBART_TO_LANG = {v: k for k, v in LANG_CODE_MAP.items()}

# Human-readable language names
LANG_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "hi": "Hindi",
    "ar": "Arabic", "pt": "Portuguese", "it": "Italian", "ru": "Russian",
    "nl": "Dutch", "tr": "Turkish", "vi": "Vietnamese"
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
    
    Supports multilingual summarization using mBART and keyword extraction
    using KeyBERT with multilingual models.
    """
    
    def __init__(
        self,
        summarizer: Optional[pipeline] = None,
        kw_model: Optional[KeyBERT] = None,
        default_chunk_size: int = 500,
        min_words_for_summary: int = 50,
        default_keyword_count: int = 10
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
        """
        self.summarizer = summarizer
        self.kw_model = kw_model
        self.default_chunk_size = default_chunk_size
        self.min_words_for_summary = min_words_for_summary
        self.default_keyword_count = default_keyword_count
        
        # Initialize translation components (lazy loading)
        self._translator_model = None
        self._translator_tokenizer = None
    
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
        Uses mBART-50 for multilingual translation.
        """
        if self._translator_model is None:
            try:
                logger.info("Loading translation model (mBART-50)...")
                model_name = "facebook/mbart-large-50-many-to-many-mmt"
                self._translator_tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
                self._translator_model = MBartForConditionalGeneration.from_pretrained(model_name)
                logger.info("✅ Translation model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load translation model: {e}")
                return None, None
        return self._translator_model, self._translator_tokenizer
    
    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate.
            source_lang: Source language code (2-letter, e.g., 'it', 'en').
            target_lang: Target language code (2-letter, e.g., 'en', 'es').
            
        Returns:
            Translated text, or original text if translation fails.
        """
        if source_lang == target_lang:
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
        
        try:
            # Set source language for tokenizer
            tokenizer.src_lang = src_mbart
            
            # Tokenize input
            encoded = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            
            # Generate translation with forced target language
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_mbart],
                max_length=1024
            )
            
            # Decode translation
            translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            logger.info(f"✅ Translated from {LANG_NAMES.get(source_lang, source_lang)} to {LANG_NAMES.get(target_lang, target_lang)}")
            return translation
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
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
        Summarize a single part/chunk of text.
        
        Args:
            chunk_text: The text to summarize.
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
            logger.info(f"📄 Summarizing {title}: {word_count} words -> target {min_length}-{max_length} words")
            
            result = self.summarizer(
                chunk_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
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
                do_sample=False
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
        Summarize text into structured parts with 3-step architecture:
        
        STEP 1: Detect source language of transcript
        STEP 2: Summarize each part in source language
        STEP 3: Translate all part summaries to output language
        
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
        # STEP 2: SUMMARIZE EACH PART IN SOURCE LANGUAGE
        # ============================================================
        logger.info(f"")
        logger.info(f"📝 STEP 2 - SUMMARIZE IN SOURCE LANGUAGE ({LANG_NAMES.get(source_lang, source_lang)})")
        
        chunks = self._chunk_text_with_positions(cleaned_text, chunk_size)
        total_parts = len(chunks)
        logger.info(f"   Divided into {total_parts} parts")
        
        # Summarize each part (in source language)
        part_summaries: List[PartSummary] = []
        
        for i, (chunk_text, start_pos, end_pos) in enumerate(chunks, 1):
            part_summary = self._summarize_single_part(
                chunk_text=chunk_text,
                part_number=i,
                total_parts=total_parts,
                start_pos=start_pos,
                end_pos=end_pos,
                compression_ratio=compression_ratio,
                extract_keywords=extract_part_keywords
            )
            logger.info(f"   ✓ Part {i}/{total_parts}: {len(part_summary.summary.split())} words in {LANG_NAMES.get(source_lang, source_lang)}")
            part_summaries.append(part_summary)
        
        # Generate overview (still in source language)
        overview_in_source = ""
        if generate_overview:
            overview_in_source = self._generate_overview(part_summaries, compression_ratio=0.5)
            logger.info(f"   ✓ Overview: {len(overview_in_source.split())} words in {LANG_NAMES.get(source_lang, source_lang)}")
        
        # ============================================================
        # STEP 3: TRANSLATE TO OUTPUT LANGUAGE (if different)
        # ============================================================
        logger.info(f"")
        logger.info(f"🔄 STEP 3 - TRANSLATE TO OUTPUT LANGUAGE")
        
        if source_lang != output_lang:
            logger.info(f"   Translating: {LANG_NAMES.get(source_lang, source_lang)} -> {LANG_NAMES.get(output_lang, output_lang)}")
            for i, part in enumerate(part_summaries, 1):
                part.summary = self._translate(part.summary, source_lang, output_lang)
                logger.info(f"   ✓ Part {i} translated")
            
            if overview_in_source:
                overview = self._translate(overview_in_source, source_lang, output_lang)
                logger.info(f"   ✓ Overview translated")
            else:
                overview = ""
        else:
            logger.info(f"   No translation needed (source = output = {LANG_NAMES.get(source_lang, source_lang)})")
            overview = overview_in_source
        
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
        # Increased max_length from 150 to 250 to allow longer summaries and reduce truncation
        max_length = max(30, min(int(input_length * compression_ratio), 250))
        min_length = min(20, int(input_length * 0.1))
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
        
        STEP 1: Transcript (source language) 
        STEP 2: Summarize (in source language)
        STEP 3: Translate summary (to user's output language)
        
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
            output_lang = source_lang  # Output in same language as transcript
            logger.info(f"   Output language: AUTO -> {LANG_NAMES.get(output_lang, output_lang)}")
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
        # STEP 2: TRANSLATE TRANSCRIPT TO ENGLISH (if needed)
        # Summarization model (BART/PEGASUS) only works with English
        # ============================================================
        text_for_summarization = cleaned_text
        summary_lang = "en"  # BART/PEGASUS always summarizes in English
        
        if source_lang != "en":
            logger.info(f"")
            logger.info(f"🔄 STEP 2A - TRANSLATE TRANSCRIPT TO ENGLISH")
            logger.info(f"   Translating from {LANG_NAMES.get(source_lang, source_lang)} -> English")
            text_for_summarization = self._translate(cleaned_text, source_lang, "en")
            logger.info(f"   ✓ Transcript translated to English ({len(text_for_summarization.split())} words)")
        else:
            logger.info(f"")
            logger.info(f"📝 STEP 2 - TRANSCRIPT ALREADY IN ENGLISH (no translation needed)")
        
        # ============================================================
        # STEP 3: SUMMARIZE IN ENGLISH USING BART/PEGASUS
        # mBART is NOT used for summarization - only for translation
        # ============================================================
        logger.info(f"")
        logger.info(f"📝 STEP 3 - SUMMARIZE IN ENGLISH (using BART/PEGASUS)")
        logger.info(f"   Summary language: {summary_lang}")
        
        chunk_size = chunk_size or self.default_chunk_size
        chunks = self._chunk_text(text_for_summarization, chunk_size)
        
        summaries = []
        
        for i, chunk in enumerate(chunks):
            try:
                chunk_words = chunk.split()
                input_len = len(chunk_words)
                
                min_length, max_length = self._calculate_summary_length(input_len, compression_ratio)
                
                logger.info(f"   Chunk {i+1}/{len(chunks)}: {input_len} words -> target {min_length}-{max_length}")
                
                # Summarize using BART/PEGASUS (English-only models, no language control needed)
                result = self.summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
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
        
        # Combine chunk summaries (in English)
        summary_in_english = " ".join(summaries)
        logger.info(f"   Combined summary: {len(summary_in_english.split())} words in English")
        logger.info(f"   Preview: {summary_in_english[:150]}...")
        
        # ============================================================
        # STEP 4: TRANSLATE SUMMARY TO OUTPUT LANGUAGE (if needed)
        # Use mBART ONLY for translation, not summarization
        # ============================================================
        logger.info(f"")
        logger.info(f"🔄 STEP 4 - TRANSLATE SUMMARY TO OUTPUT LANGUAGE")
        
        if output_lang != "en":
            logger.info(f"   Translating: English -> {LANG_NAMES.get(output_lang, output_lang)}")
            final_summary = self._translate(summary_in_english, "en", output_lang)
            logger.info(f"   ✓ Translation complete: {len(final_summary.split())} words")
            logger.info(f"   Preview: {final_summary[:150]}...")
        else:
            logger.info(f"   No translation needed (output = English)")
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
