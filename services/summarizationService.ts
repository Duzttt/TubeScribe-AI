/**
 * Service for calling the Python FastAPI summarization backend
 * Provides multilingual summarization and keyword extraction
 */

export interface SummarizeRequest {
  text: string;
  lang?: string; // Language code like "en_XX", "zh_CN", "es_XX", etc.
}

export interface SummarizeResponse {
  summary: string;
  keywords: string[];
  source_language?: string;
  target_language?: string;
}

// Backend URL configuration for Python summarization service
const SUMMARIZATION_BACKEND_URL = process.env.SUMMARIZATION_BACKEND_URL || "http://localhost:8000";

/**
 * Maps TargetLanguage enum to mBART language codes
 */
const getLanguageCode = (targetLanguage: string): string => {
  const langMap: Record<string, string> = {
    'English': 'en_XX',
    'Spanish': 'es_XX',
    'French': 'fr_XX',
    'German': 'de_DE',
    'Chinese (Simplified)': 'zh_CN',
    'Japanese': 'ja_XX',
    'Korean': 'ko_KR',
    'Hindi': 'hi_IN',
    'Arabic': 'ar_AR',
    'Portuguese': 'pt_XX',
    'Italian': 'it_IT',
    'Russian': 'ru_RU',
    'Dutch': 'nl_XX',
    'Turkish': 'tr_TR',
    'Vietnamese': 'vi_VN',
  };
  
  return langMap[targetLanguage] || 'en_XX';
};

/**
 * Checks if the Python summarization backend is available
 */
export const checkBackendHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${SUMMARIZATION_BACKEND_URL}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) return false;
    const data = await response.json();
    // Check if status is healthy and summary_service is ready
    return data.status === 'healthy' && 
           data.models_ready && 
           (data.models_ready.summary_service === true || 
            (data.models_ready.summarizer === true && data.models_ready.keyword_extractor === true));
  } catch (error) {
    console.warn('Summarization backend health check failed:', error);
    return false;
  }
};

/**
 * Summarizes text and extracts keywords using the Python backend
 * 
 * @param text - The text to summarize
 * @param targetLanguage - Target language for summarization (defaults to English)
 * @returns Promise with summary and keywords
 */
export const summarizeWithKeywords = async (
  text: string,
  targetLanguage: string | null = 'English'
): Promise<SummarizeResponse> => {
  if (!text || !text.trim()) {
    throw new Error('Text cannot be empty');
  }

  // If null, use special code "AUTO" to let backend detect and use transcript language
  const langCode = targetLanguage === null ? 'AUTO' : getLanguageCode(targetLanguage);
  
  const requestBody: SummarizeRequest = {
    text: text.trim(),
    lang: langCode,
  };

  try {
    const controller = new AbortController();
    // Increased timeout to 10 minutes for the new flow: translate → summarize → translate
    // Long transcripts with translation steps can take 5-10 minutes
    const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 minute timeout

    try {
      const response = await fetch(`${SUMMARIZATION_BACKEND_URL}/summarize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `Backend error (${response.status}): ${response.statusText}`);
      }

      const data: SummarizeResponse = await response.json();
      
      // Validate response structure
      if (!data.summary || !Array.isArray(data.keywords)) {
        throw new Error('Invalid response format from summarization backend');
      }

      return data;
    } catch (fetchError) {
      clearTimeout(timeoutId);
      if (fetchError instanceof Error && fetchError.name === 'AbortError') {
        throw new Error('Summarization request timed out after 10 minutes');
      }
      throw fetchError;
    }
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    console.error('Summarization service error:', errorMsg);
    throw new Error(`Failed to summarize text: ${errorMsg}`);
  }
};

/**
 * Get backend status information
 */
export const getBackendStatus = async (): Promise<{
  status: string;
  models_loaded: { summarizer: boolean; keyword_extractor: boolean };
} | null> => {
  try {
    const response = await fetch(`${SUMMARIZATION_BACKEND_URL}/`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    console.warn('Failed to get backend status:', error);
    return null;
  }
};