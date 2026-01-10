const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const youtubedl = require('youtube-dl-exec');
const { YoutubeTranscript } = require('youtube-transcript');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { GoogleAIFileManager } = require('@google/generative-ai/server');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 7860;

const fileManager = new GoogleAIFileManager(process.env.GEMINI_API_KEY);
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

const tempDir = path.join(__dirname, 'temp');
if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir);
}

const formatTimestamp = (offsetMs) => {
    const totalSeconds = Math.floor(offsetMs / 1000);
    const mm = Math.floor(totalSeconds / 60).toString().padStart(2, '0');
    const ss = (totalSeconds % 60).toString().padStart(2, '0');
    return `[${mm}:${ss}]`;
};

app.get('/', (req, res) => {
    res.send('TubeScribe Backend is Running (v3: Anti-Bot Headers)');
});

app.post('/api/transcribe', async (req, res) => {
    const { videoUrl, targetLanguage } = req.body;
    const timestamp = Date.now();
    const filename = `audio_${timestamp}.mp3`;
    const filePath = path.join(tempDir, filename);
    const cookiesPath = path.join(__dirname, 'cookies.txt'); // Check for cookies file

    console.log(`[${timestamp}] Processing: ${videoUrl}`);

    // =======================================================
    // STRATEGY 1: INSTANT CAPTION EXTRACTION
    // =======================================================
    try {
        console.log("Strategy 1: Attempting to fetch existing captions...");
        const captionItems = await YoutubeTranscript.fetchTranscript(videoUrl);

        console.log(`✅ Success! Found ${captionItems.length} lines.`);
        let fullTranscript = captionItems.map(item => `${formatTimestamp(item.offset)} ${item.text}`).join('\n');

        if (targetLanguage !== 'Original') {
            const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
            const result = await model.generateContent(`Translate to ${targetLanguage}. Keep [MM:SS].\n\n${fullTranscript.substring(0, 30000)}`);
            fullTranscript = result.response.text();
        }
        return res.json({ transcript: fullTranscript });

    } catch (error) {
        console.warn(`⚠️ Caption extraction failed (${error.message}). Falling back...`);
    }

    // =======================================================
    // STRATEGY 2: AUDIO DOWNLOAD (With Anti-Bot Measures)
    // =======================================================
    try {
        console.log("Strategy 2: Downloading audio with yt-dlp...");

        // Define flags to mimic a real browser
        const ytdlFlags = {
            extractAudio: true,
            audioFormat: 'mp3',
            output: filePath,
            noCheckCertificates: true,
            noWarnings: true,
            preferFreeFormats: true,
            // Mimic a real Chrome browser on Windows
            userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            referer: 'https://www.youtube.com/',
        };

        // If you uploaded a cookies.txt file, use it (Nuclear Option)
        if (fs.existsSync(cookiesPath)) {
            console.log("wb Using cookies.txt for authentication");
            ytdlFlags.cookies = cookiesPath;
        }

        await youtubedl(videoUrl, ytdlFlags);

        if (!fs.existsSync(filePath)) throw new Error("File not found.");
        const stats = fs.statSync(filePath);
        console.log(`Downloaded size: ${stats.size} bytes`);

        if (stats.size < 10000) throw new Error("File too small (likely blocked).");

        console.log("Uploading to Gemini...");
        const uploadResult = await fileManager.uploadFile(filePath, { mimeType: "audio/mp3", displayName: `Audio_${timestamp}` });

        let file = await fileManager.getFile(uploadResult.file.name);
        while (file.state === "PROCESSING") {
            await new Promise(r => setTimeout(r, 2000));
            file = await fileManager.getFile(uploadResult.file.name);
        }

        if (file.state === "FAILED") throw new Error("Gemini processing failed.");

        const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
        const prompt = `Transcribe this audio exactly. Format: "[MM:SS] Speaker: Text". ${targetLanguage !== 'Original' ? `Translate to ${targetLanguage}.` : ''}`;

        const result = await model.generateContent([{ fileData: { mimeType: uploadResult.file.mimeType, fileUri: uploadResult.file.uri } }, { text: prompt }]);

        if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
        res.json({ transcript: result.response.text() });

    } catch (error) {
        console.error("❌ Strategy 2 Failed:", error);
        if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
        // Pass the actual error message back to frontend for better debugging
        res.status(500).json({ error: `Transcription failed. YouTube blocked the download. Details: ${error.message}` });
    }
});

app.listen(PORT, () => {
    console.log(`Backend running on port ${PORT}`);
});