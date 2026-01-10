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
    res.send('TubeScribe Backend Running (v7: Mobile Agent + Direct URL)');
});

app.post('/api/transcribe', async (req, res) => {
    const { videoUrl, targetLanguage } = req.body;
    const timestamp = Date.now();
    const filename = `audio_${timestamp}.mp3`;
    const filePath = path.join(tempDir, filename);

    console.log(`[${timestamp}] Processing: ${videoUrl}`);

    // =======================================================
    // STRATEGY 1: INSTANT CAPTION EXTRACTION
    // =======================================================
    try {
        console.log("Strategy 1: Attempting to fetch existing captions...");

        // We use your exact URL. youtube-transcript handles most formats well.
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
        console.warn(`⚠️ Strategy 1 Failed: ${error.message}`);
        console.log("Falling back to Strategy 2 (Audio Download)...");
    }

    // =======================================================
    // STRATEGY 2: AUDIO DOWNLOAD (Android Emulation)
    // =======================================================
    try {
        console.log("Strategy 2: Downloading audio via yt-dlp...");

        await youtubedl(videoUrl, {
            extractAudio: true,
            audioFormat: 'mp3',
            output: filePath,
            // SECURITY BYPASS FLAGS
            noCheckCertificates: true,
            noWarnings: true,
            preferFreeFormats: true,
            forceIpv4: true, // Fixes [Errno -5]

            // KEY FIX: Use Android Mobile User Agent
            // YouTube rarely blocks "Mobile" traffic from data centers compared to "Desktop"
            userAgent: 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36',
            addHeader: ['referer:m.youtube.com']
        });

        if (!fs.existsSync(filePath)) throw new Error("File not found.");
        const stats = fs.statSync(filePath);

        // If file is too small, it failed
        if (stats.size < 10000) throw new Error("YouTube blocked the download (IP Reputation).");

        console.log("Uploading to Gemini...");
        const uploadResult = await fileManager.uploadFile(filePath, { mimeType: "audio/mp3", displayName: `Audio_${timestamp}` });

        let file = await fileManager.getFile(uploadResult.file.name);
        while (file.state === "PROCESSING") {
            await new Promise(r => setTimeout(r, 2000));
            file = await fileManager.getFile(uploadResult.file.name);
        }

        if (file.state === "FAILED") throw new Error("Gemini processing failed.");

        const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
        const prompt = `Transcribe exactly. Format: "[MM:SS] Speaker: Text". ${targetLanguage !== 'Original' ? `Translate to ${targetLanguage}.` : ''}`;

        const result = await model.generateContent([{ fileData: { mimeType: uploadResult.file.mimeType, fileUri: uploadResult.file.uri } }, { text: prompt }]);

        if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
        res.json({ transcript: result.response.text() });

    } catch (error) {
        console.error("❌ Strategy 2 Failed:", error);
        if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
        res.status(500).json({
            error: "Could not transcribe. YouTube blocked the connection from this server."
        });
    }
});

app.listen(PORT, () => {
    console.log(`Backend running on port ${PORT}`);
});