const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
// The stable audio downloader
const youtubedl = require('youtube-dl-exec');
// The instant caption extractor
const { YoutubeTranscript } = require('youtube-transcript');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { GoogleAIFileManager } = require('@google/generative-ai/server');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 7860;

// Initialize Gemini
const fileManager = new GoogleAIFileManager(process.env.GEMINI_API_KEY);
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Ensure temp directory exists
const tempDir = path.join(__dirname, 'temp');
if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir);
}

// Helper to format timestamps [MM:SS]
const formatTimestamp = (offsetMs) => {
    const totalSeconds = Math.floor(offsetMs / 1000);
    const mm = Math.floor(totalSeconds / 60).toString().padStart(2, '0');
    const ss = (totalSeconds % 60).toString().padStart(2, '0');
    return `[${mm}:${ss}]`;
};

app.get('/', (req, res) => {
    res.send('TubeScribe Backend is Running (Strategy 1.5 + 2)');
});

app.post('/api/transcribe', async (req, res) => {
    const { videoUrl, targetLanguage } = req.body;
    const timestamp = Date.now();
    const filename = `audio_${timestamp}.mp3`;
    const filePath = path.join(tempDir, filename);

    console.log(`[${timestamp}] Processing: ${videoUrl}`);

    // =======================================================
    // STRATEGY 1: INSTANT CAPTION EXTRACTION (Preferred)
    // =======================================================
    try {
        console.log("Strategy 1: Attempting to fetch existing captions...");

        // Fetch raw captions
        const captionItems = await YoutubeTranscript.fetchTranscript(videoUrl);

        console.log(`✅ Success! Found ${captionItems.length} lines of captions.`);

        // Format them nicely
        let fullTranscript = captionItems.map(item => {
            return `${formatTimestamp(item.offset)} ${item.text}`;
        }).join('\n');

        // If user wants translation, use Gemini here (It's cheap/fast on text)
        if (targetLanguage !== 'Original') {
            console.log(`Translating text to ${targetLanguage}...`);
            const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
            const prompt = `
                Translate the following transcript to ${targetLanguage}.
                Keep the [MM:SS] timestamps exactly as they are at the start of lines.
                Do not summarize. 
                
                TRANSCRIPT:
                ${fullTranscript.substring(0, 30000)}
            `;
            const result = await model.generateContent(prompt);
            fullTranscript = result.response.text();
        }

        return res.json({ transcript: fullTranscript });

    } catch (error) {
        console.warn(`⚠️ Caption extraction unavailable (${error.message}). Falling back to audio download...`);
    }

    // =======================================================
    // STRATEGY 2: AUDIO DOWNLOAD + GEMINI LISTENING (Fallback)
    // =======================================================
    try {
        console.log("Strategy 2: Downloading audio with yt-dlp...");

        // Download as MP3
        await youtubedl(videoUrl, {
            extractAudio: true,
            audioFormat: 'mp3',
            output: filePath,
            noCheckCertificates: true,
            noWarnings: true,
            preferFreeFormats: true,
            addHeader: ['referer:youtube.com', 'user-agent:googlebot']
        });

        // Verify download success
        if (!fs.existsSync(filePath)) throw new Error("Download failed: File not found.");
        const stats = fs.statSync(filePath);
        console.log(`Downloaded file size: ${stats.size} bytes`);

        // If file is too small (<10KB), YouTube likely blocked us
        if (stats.size < 10000) {
            throw new Error("Download failed: File too small (YouTube blocked IP).");
        }

        console.log("Uploading to Gemini...");
        const uploadResult = await fileManager.uploadFile(filePath, {
            mimeType: "audio/mp3",
            displayName: `Audio_${timestamp}`,
        });

        // Wait for Gemini to process audio
        let file = await fileManager.getFile(uploadResult.file.name);
        while (file.state === "PROCESSING") {
            await new Promise((resolve) => setTimeout(resolve, 2000));
            file = await fileManager.getFile(uploadResult.file.name);
        }

        if (file.state === "FAILED") {
            throw new Error("Gemini failed to process the audio file.");
        }

        console.log("Generating transcript from audio...");
        const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
        const isOriginal = targetLanguage === 'Original';
        const prompt = `
            Transcribe this audio file exactly.
            ${isOriginal ? "Return original language." : `Translate to ${targetLanguage}.`}
            Format: "[MM:SS] Speaker: Text"
        `;

        const result = await model.generateContent([
            {
                fileData: {
                    mimeType: uploadResult.file.mimeType,
                    fileUri: uploadResult.file.uri
                }
            },
            { text: prompt }
        ]);

        const transcript = result.response.text();

        // Cleanup
        if (fs.existsSync(filePath)) fs.unlinkSync(filePath);

        res.json({ transcript });

    } catch (error) {
        console.error("❌ All strategies failed:", error);
        if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
        res.status(500).json({ error: "Failed to transcribe video. No captions found and audio download blocked." });
    }
});

app.listen(PORT, () => {
    console.log(`Backend running on port ${PORT}`);
});