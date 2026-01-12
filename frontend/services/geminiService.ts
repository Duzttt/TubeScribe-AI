import { GoogleGenAI } from "@google/genai";
import { ChatMessage, TargetLanguage } from '../types';

const getAiClient = () => {
  const apiKey = process.env.API_KEY;
  console.log("🔑 [GEMINI] Checking API key:", apiKey ? "Found" : "Missing");
  if (!apiKey) {
    console.error("❌ [GEMINI] API_KEY environment variable is missing.");
    throw new Error("API_KEY environment variable is missing.");
  }
  return new GoogleGenAI({ apiKey });
};

// Using gemini-3-flash-preview for text tasks as per guidelines
const TEXT_MODEL = "gemini-3-flash-preview";

// Backend URL configuration
// Python backend for transcription (using local Whisper model)
const PYTHON_BACKEND_URL = process.env.VITE_PYTHON_BACKEND_URL || "http://localhost:8000";

/**
 * Helper function to add timeout to a promise
 */
const withTimeout = <T>(promise: Promise<T>, timeoutMs: number): Promise<T> => {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(`Operation timed out after ${timeoutMs}ms`)), timeoutMs)
    )
  ]);
};

/**
 * Generates a transcript from a YouTube URL using local Whisper model.
 * Uses Python backend with Whisper model for transcription (no API keys needed).
 */
export const generateTranscript = async (
  videoUrl: string,
  targetLanguage: TargetLanguage = TargetLanguage.ORIGINAL
): Promise<string> => {
  console.log("🚀 [TRANSCRIPT] Starting transcription process...");
  console.log("📹 [TRANSCRIPT] Video URL:", videoUrl);
  console.log("🌍 [TRANSCRIPT] Target Language:", targetLanguage);

  // Check if Python backend is available
  console.log("🔍 [TRANSCRIPT] Checking Python backend health...");
  const healthCheck = await fetch(`${PYTHON_BACKEND_URL}/health`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  }).catch(() => null);

  if (!healthCheck || !healthCheck.ok) {
    console.error("❌ [TRANSCRIPT] Python backend health check failed");
    throw new Error(`Python backend is not available at ${PYTHON_BACKEND_URL}. Please make sure the Python backend is running.`);
  }

  const healthData = await healthCheck.json();
  if (!healthData.models_ready?.transcriber) {
    console.error("❌ [TRANSCRIPT] Transcription model not loaded in backend");
    throw new Error('Python backend transcription model is not loaded. Please check the backend logs.');
  }

  console.log("✅ [TRANSCRIPT] Python backend is healthy and ready");
  console.log("📤 [TRANSCRIPT] Sending transcription request to backend...");

  // Use Python backend for transcription
  const controller = new AbortController();
  // Longer timeout for transcription (5 minutes for long videos)
  const timeoutId = setTimeout(() => controller.abort(), 300000);

  try {
    const startTime = Date.now();
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/transcribe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        videoUrl,
        targetLanguage: targetLanguage === TargetLanguage.ORIGINAL ? null : targetLanguage
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(1);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      console.error("❌ [TRANSCRIPT] Backend returned error:", errorData);
      throw new Error(errorData.detail || `Python backend error (${response.status})`);
    }

    const data = await response.json();
    console.log(`✅ [TRANSCRIPT] Transcript received! (${elapsedTime}s)`);
    console.log(`📝 [TRANSCRIPT] Transcript length: ${data.transcript.length} characters`);
    
    // If translation is needed and not original, translate using Gemini
    if (targetLanguage !== TargetLanguage.ORIGINAL && data.transcript) {
      console.log(`🌍 [TRANSCRIPT] Translation needed to: ${targetLanguage}`);
      console.log("🔄 [TRANSCRIPT] Starting translation...");
      const translateResult = await translateContent(data.transcript, targetLanguage);
      console.log("✅ [TRANSCRIPT] Translation completed");
      return translateResult.translatedText;
    }
    
    console.log("✅ [TRANSCRIPT] Process completed successfully");
    return data.transcript;
  } catch (fetchError) {
    clearTimeout(timeoutId);
    if (fetchError instanceof Error && fetchError.name === 'AbortError') {
      console.error("⏱️ [TRANSCRIPT] Request timed out after 5 minutes");
      throw new Error('Transcription request timed out after 5 minutes. The video may be too long or the backend is slow.');
    }
    console.error("❌ [TRANSCRIPT] Error:", fetchError);
    if (fetchError instanceof Error) {
      throw fetchError;
    }
    throw new Error(`Failed to generate transcript: ${String(fetchError)}`);
  }
};


/**
 * Translates content using LOCAL Python backend (mBART-50 model).
 * 100% LOCAL - NO API KEYS REQUIRED!
 * Uses GPU if available for fast translation.
 * 
 * If video_id is provided, uses cached transcript from transcription instead of text parameter.
 * This is more efficient as it avoids sending large transcript text over the network.
 * 
 * Supports real-time progress tracking via progressCallback. If provided, the function will
 * connect to the SSE stream and call the callback with progress updates.
 * 
 * @param text - Text to translate (required if video_id not provided)
 * @param targetLanguage - Target language for translation
 * @param video_id - Optional video ID to use cached transcript instead of text
 * @param progressCallback - Optional callback function to receive progress updates (progress: number, message: string)
 * @returns Promise that resolves to the translated text and the request_id for progress tracking
 */
export const translateContent = async (
  text: string,
  targetLanguage: TargetLanguage,
  video_id?: string,
  progressCallback?: (progress: number, message: string) => void
): Promise<{ translatedText: string; requestId?: string }> => {
  if (targetLanguage === TargetLanguage.ORIGINAL) {
    return { translatedText: text }; // No op
  }

  // Map TargetLanguage enum values to language codes for mBART
  const languageMap: Record<string, string> = {
    [TargetLanguage.ENGLISH]: "en",
    [TargetLanguage.CHINESE_SIMPLIFIED]: "zh",
    [TargetLanguage.SPANISH]: "es",
    [TargetLanguage.FRENCH]: "fr",
    [TargetLanguage.GERMAN]: "de",
    [TargetLanguage.JAPANESE]: "ja",
    [TargetLanguage.KOREAN]: "ko",
    [TargetLanguage.PORTUGUESE]: "pt",
    [TargetLanguage.ITALIAN]: "it",
    [TargetLanguage.RUSSIAN]: "ru",
    [TargetLanguage.HINDI]: "hi",
    [TargetLanguage.ARABIC]: "ar",
    [TargetLanguage.DUTCH]: "nl",
    [TargetLanguage.TURKISH]: "tr",
    [TargetLanguage.VIETNAMESE]: "vi",
    [TargetLanguage.MALAY]: "ms",
    [TargetLanguage.ORIGINAL]: "auto"
  };

  const targetLangCode = languageMap[targetLanguage] || "en";

  if (video_id) {
    console.log(`🔄 [TRANSLATE] Using LOCAL Python backend (mBART-50) with cached transcript (video_id: ${video_id}) for translation to ${targetLanguage}...`);
  } else {
    console.log(`🔄 [TRANSLATE] Using LOCAL Python backend (mBART-50) for translation to ${targetLanguage}...`);
  }

  try {
    // Generate request ID for progress tracking
    const requestId = progressCallback ? `translate_${Date.now()}_${Math.random().toString(36).substr(2, 9)}` : undefined;

    const requestBody: any = {
      target_language: targetLangCode,
      source_language: null,  // Auto-detect
      request_id: requestId
    };

    // If video_id is provided, use cached transcript; otherwise use text
    if (video_id) {
      requestBody.video_id = video_id;
      // text is optional when video_id is provided
    } else {
      requestBody.text = text;
    }

    // Connect to progress stream if callback provided
    let eventSource: EventSource | null = null;
    if (progressCallback && requestId) {
      try {
        eventSource = new EventSource(`${PYTHON_BACKEND_URL}/api/translate/${requestId}/stream`);
        
        eventSource.onmessage = (event) => {
          try {
            const progress = JSON.parse(event.data);
            const progressValue = progress.progress || progress.translation_progress || 0;
            const message = progress.message || 'Translating...';
            progressCallback(progressValue, message);
            
            // Close connection if completed or error
            if (progress.status === 'completed' || progress.status === 'error') {
              eventSource?.close();
              eventSource = null;
            }
          } catch (error) {
            console.warn('Failed to parse translation progress:', error);
          }
        };
        
        eventSource.onerror = (error) => {
          console.warn('Translation progress SSE connection error:', error);
          eventSource?.close();
          eventSource = null;
        };
      } catch (error) {
        console.warn('Failed to connect to translation progress stream:', error);
        // Continue without progress tracking
      }
    }

    const response = await fetch(`${PYTHON_BACKEND_URL}/api/translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      console.error("❌ [TRANSLATE] Backend returned error:", errorData);
      eventSource?.close();
      throw new Error(errorData.detail || `Translation failed (${response.status})`);
    }

    const data = await response.json();
    console.log(`✅ [TRANSLATE] Translation complete: ${data.original_length} words -> ${data.translated_length} words`);
    console.log(`   From: ${data.source_language} -> To: ${data.target_language}`);
    
    // Ensure progress is at 100%
    if (progressCallback) {
      progressCallback(100, 'Translation completed');
      // Close connection after a short delay
      setTimeout(() => {
        eventSource?.close();
      }, 500);
    }
    
    return {
      translatedText: data.translated_text || text,
      requestId: requestId
    };
  } catch (error) {
    console.error("❌ [TRANSLATE] Error translating content:", error);
    throw new Error(`Failed to translate content: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
};

/**
 * Sends a chat message with video context.
 */
export const sendChatMessage = async (
  history: ChatMessage[],
  newMessage: string,
  videoUrl: string,
  context?: string
): Promise<string> => {
  const ai = getAiClient();

  const prompt = `
    You are an intelligent video assistant.
    
    Video URL: ${videoUrl}
    
    CONTEXT (Transcript/Summary): 
    ${context ? context.substring(0, 20000) : "Context not available. Use Google Search to find information about the video."}
    
    CHAT HISTORY:
    ${history.map(msg => `${msg.role === 'user' ? 'User' : 'Model'}: ${msg.text}`).join('\n')}
    
    USER QUESTION: ${newMessage}
    
    INSTRUCTIONS:
    1. Answer the user's question accurately based on the video context.
    2. **TIMESTAMPS**: You MUST include timestamps (e.g., [02:30], [10:15]) when referencing specific events, quotes, or details from the video. 
       - If timestamps are present in the provided transcript/context, use them.
       - If not, use the 'googleSearch' tool to find the approximate time of specific events in the video.
    3. Keep the response helpful, conversational, and easy to read.
  `;

  try {
    const response = await ai.models.generateContent({
      model: TEXT_MODEL,
      contents: prompt,
      config: {
        tools: [{ googleSearch: {} }],
      }
    });

    return response.text || "I couldn't generate a response.";
  } catch (error) {
    console.error("Error in chat:", error);
    return "Sorry, I encountered an error processing your request.";
  }
};

/**
 * Sends a chat message to the general app guide.
 */
export const sendGuideMessage = async (
  history: ChatMessage[],
  newMessage: string
): Promise<string> => {
  const ai = getAiClient();

  const systemInstruction = `You are the friendly and intelligent Project Guide for "TubeScribe AI".
  Your purpose is to assist users in understanding and navigating this web application.

  **Capabilities of TubeScribe AI:**
  1. **Transcription:** Converts YouTube video speech into text with timestamps.
  2. **Summarization:** Creates concise summaries and key takeaways from videos.
  3. **Translation:** Translates transcripts or summaries into 15+ languages.
  4. **Chat:** Allows users to ask questions about the video content interactively.

  **Common User Queries (Handle these efficiently):**
  1. *What is this project?* -> It's an AI-powered video analysis tool using Google Gemini 3 flash preview.
  2. *How do I use it?* -> Paste a YouTube link, choose the language then click Transcribe or Summarize.
  3. *Is it free?* -> Yes, this is a demo application showing the power of the Gemini API.
  4. *Can it download videos?* -> No, it analyzes the content but doesn't provide video file downloads.
  5. *What languages are supported?* -> English, Spanish, French, German, Chinese, Japanese, and many more.
  6. *How accurate is it?* -> It uses Gemini 3.0 Flash Preview, which has state-of-the-art multimodal understanding.
  7. *How long does it take?* -> Usually a few minutes, depending on video length and server load.
  8. *Does it work on long videos?* -> Yes, Gemini has a large context window suitable for long content for it need some time.
  9. *Privacy?* -> We do not store your data permanently; processing is done on the fly.
  10. *Error handling?* -> If a video is private or age-restricted, the AI might not be able to access it.
  11. *Can I ask questions about the video?* -> Yes, use the Chat feature after transcription/summarization.
  


  **Navigation Protocol:**
  If the user says "Start", "Open the app", "Let's go", "I want to transcribe", or indicates they are ready to use the tool, you MUST append the tag **[ACTION:START_APP]** to the end of your response. This will automatically redirect them to the main interface.

  **Tone:**
  Professional, helpful, and concise.
  `;

  let prompt = `Chat History:\n`;
  history.forEach(msg => {
    prompt += `${msg.role === 'user' ? 'User' : 'Guide'}: ${msg.text}\n`;
  });
  prompt += `\nUser: ${newMessage}\nGuide:`;

  try {
    console.log("🚀 [GEMINI] Sending guide message to Gemini...");
    const response = await ai.models.generateContent({
      model: TEXT_MODEL,
      contents: prompt,
      config: {
        systemInstruction: systemInstruction
      }
    });

    console.log("✅ [GEMINI] Received response from Gemini");
    return response.text || "I'm here to help you get started!";
  } catch (error) {
    console.error("❌ [GEMINI] Error in guide chat:", error);
    if (error instanceof Error) {
      console.error("❌ [GEMINI] Error message:", error.message);
      console.error("❌ [GEMINI] Error stack:", error.stack);
    }
    return "I'm having trouble connecting right now, but feel free to just click 'Start Analyzing' to try the app!";
  }
};