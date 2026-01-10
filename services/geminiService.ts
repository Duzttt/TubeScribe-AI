import { GoogleGenAI } from "@google/genai";
import { TargetLanguage, ChatMessage } from '../types';

const getAiClient = () => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    throw new Error("API_KEY environment variable is missing.");
  }
  return new GoogleGenAI({ apiKey });
};

// Using gemini-3-flash-preview for text tasks as per guidelines
const TEXT_MODEL = "gemini-3-flash-preview";

const BACKEND_URL = "https://kaiwen03-tubescribe-backend.hf.space";
/**
 * Generates a transcript from a YouTube URL.
 * HYBRID STRATEGY:
 * 1. Tries to find existing captions via Google Search (Fast).
 * 2. If no captions found, falls back to Backend Audio Download (Slow but guaranteed).
 */
export const generateTranscript = async (
  videoUrl: string,
  targetLanguage: TargetLanguage = TargetLanguage.ORIGINAL
): Promise<string> => {

  const ai = getAiClient();

  const isOriginal = targetLanguage === TargetLanguage.ORIGINAL;

  // --- STEP 1: Try Frontend Search (Captions) ---
  try {
    console.log("Strategy 1: Attempting to find captions via Google Search...");

    const searchPrompt = `
      Video URL: ${videoUrl}
   
      You are an expert Video Transcriber and Translator.
    
      TASK:
      Generate a text transcript of the spoken content in this video.
    
      INSTRUCTIONS:
      1. **SEARCH**: Use 'googleSearch' to find the official captions or transcript for this video.
      2. If you find a transcript, ensure it matches the video title exactly.
      3. **SPEECH TO TEXT & AUDIO ONLY**: Listen STRICTLY to the spoken audio track.
      4. **DO NOT GENERATE TEXT YOURSELF.** If you cannot find an existing transcript on the web, you MUST return the string "NO_TRANSCRIPT_FOUND".
      5. **FORMAT**:
        - **INCLUDE TIMESTAMPS** (e.g., [00:30], [01:45]) for every new speaker or significant segment. This is crucial for navigation.
        - Format as: "**[Timestamp] Speaker:** Text" (Use double newlines between speakers for clarity).
      6. **IGNORE METADATA**: Do NOT include text from the YouTube video description, video title, or tags.  
      7. **LANGUAGE & TRANSLATION**:
        ${isOriginal
        ? `- Output the transcript in the **ORIGINAL SPOKEN LANGUAGE**. Do not translate it.`
        : `- **TRANSLATE** the entire transcript into **${targetLanguage}**.`
      }

      CONSTRAINTS:
      - If the video is music, transcribe the lyrics.
    `;


    const searchResponse = await ai.models.generateContent({
      model: TEXT_MODEL,
      contents: searchPrompt,
      config: {
        tools: [{ googleSearch: {} }],
        temperature: 0.1, // Low temp to prevent hallucination
      }
    });

    const text = searchResponse.text;

    // If we got a valid transcript (not the error code, and reasonable length)
    if (text && !text.includes("NO_TRANSCRIPT_FOUND") && text.length > 100) {
      console.log("✅ Captions found via Search.");
      return text;
    }

    console.warn("⚠️ No captions found via Search. Falling back to Backend...");

  } catch (error) {
    console.warn("⚠️ Frontend Search failed or error occurred. Falling back to Backend...", error);
  }

  // --- STEP 2: Backend Fallback (Audio Processing) ---
  try {
    console.log("Strategy 2: Calling Backend to download and transcribe audio...");

    if (BACKEND_URL.includes("YOUR-SPACE-NAME")) {
      throw new Error("Backend URL not configured in geminiService.ts");
    }

    const response = await fetch(BACKEND_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ videoUrl, targetLanguage }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || 'Backend transcription failed');
    }

    const data = await response.json();
    console.log("✅ Audio successfully transcribed by Backend.");
    return data.transcript;

  } catch (backendError) {
    console.error("❌ Both strategies failed:", backendError);
    throw new Error("Failed to generate transcript. Video has no captions and Backend could not process audio.");
  }
};

/**
 * Generates a summary from the YouTube URL in the target language.
 */
export const generateSummary = async (
  videoUrl: string,
  transcriptContext?: string,
  targetLanguage: TargetLanguage = TargetLanguage.ORIGINAL
): Promise<string> => {
  const ai = getAiClient();

  const langInstruction = targetLanguage === TargetLanguage.ORIGINAL
    ? "in the video's original language"
    : `in ${targetLanguage}`;

  let prompt = `
    Analyze this YouTube video: ${videoUrl}
    
    Task: Create a comprehensive and structured summary ${langInstruction}.
    
    Structure:
    - **TL;DR**: A 2-sentence overview.
    - **Key Takeaways**: Bullet points of the most important information.
    - **Detailed Analysis**: A deeper dive into the content.
    
    Constraints:
    - Use Google Search to verify the video content.
    - If a transcript is provided below, use it as the primary source.
    - Reference timestamps in Key Takeaways if available in the transcript.
  `;

  if (transcriptContext) {
    prompt += `\n\nTRANSCRIPT CONTEXT:\n${transcriptContext.substring(0, 25000)}`;
  }

  try {
    const response = await ai.models.generateContent({
      model: TEXT_MODEL,
      contents: prompt,
      config: {
        tools: [{ googleSearch: {} }],
      }
    });

    return response.text || "No summary generated.";
  } catch (error) {
    console.error("Error generating summary:", error);
    throw new Error("Failed to generate summary.");
  }
};

/**
 * Translates content.
 */
export const translateContent = async (
  text: string,
  targetLanguage: TargetLanguage
): Promise<string> => {
  const ai = getAiClient();

  if (targetLanguage === TargetLanguage.ORIGINAL) {
    return text; // No op
  }

  const prompt = `
    You are a professional translator.
    Task: Translate the following text into ${targetLanguage}.
    
    Guidelines:
    - Maintain the original tone and style.
    - **CRITICAL**: Preserve all timestamps (e.g., [10:05]) exactly where they appear.
    - **FORMAT**: Use Markdown. Ensure timestamps and speaker labels at the start of lines are **bold** (e.g., **[10:05] Speaker:**) to maintain a structured transcript format.
    - Preserve speaker names.
    - Do not summarize; provide a full translation.
    
    Text to translate:
    ${text.substring(0, 30000)}
  `;

  try {
    const response = await ai.models.generateContent({
      model: TEXT_MODEL,
      contents: prompt,
    });

    return response.text || "No translation generated.";
  } catch (error) {
    console.error("Error translating content:", error);
    throw new Error("Failed to translate content.");
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
  1. *What is this project?* -> It's an AI-powered video analysis tool using Google Gemini 2.0.
  2. *How do I use it?* -> Paste a YouTube link, then click Transcribe or Summarize.
  3. *Is it free?* -> Yes, this is a demo application showing the power of the Gemini API.
  4. *Can it download videos?* -> No, it analyzes the content but doesn't provide video file downloads.
  5. *What languages are supported?* -> English, Spanish, French, German, Chinese, Japanese, and many more.
  6. *How accurate is it?* -> It uses Gemini 2.0 Flash, which has state-of-the-art multimodal understanding.
  7. *Who made this?* -> This is a showcase of the Google GenAI SDK.
  8. *Does it work on long videos?* -> Yes, Gemini has a large context window suitable for long content.
  9. *Privacy?* -> We do not store your data permanently; processing is done on the fly.
  10. *Error handling?* -> If a video is private or age-restricted, the AI might not be able to access it.

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
    const response = await ai.models.generateContent({
      model: TEXT_MODEL,
      contents: prompt,
      config: {
        systemInstruction: systemInstruction
      }
    });

    return response.text || "I'm here to help you get started!";
  } catch (error) {
    console.error("Error in guide chat:", error);
    return "I'm having trouble connecting right now, but feel free to just click 'Start Analyzing' to try the app!";
  }
};