const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const ytdl = require('@distube/ytdl-core');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { GoogleAIFileManager } = require('@google/generative-ai/server');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

// --- CRITICAL CHANGE FOR HUGGING FACE ---
const PORT = 7860;
// ----------------------------------------

// Initialize Gemini
// Note: On HF, we will use Secrets, so process.env.GEMINI_API_KEY will work automatically
const fileManager = new GoogleAIFileManager(process.env.GEMINI_API_KEY);
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Ensure temp directory exists
const tempDir = path.join(__dirname, 'temp');
if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir);
}

app.get('/', (req, res) => {
    res.send('TubeScribe Backend is Running!');
});

app.post('/api/transcribe', async (req, res) => {
    const { videoUrl, targetLanguage } = req.body;
    const timestamp = Date.now();
    const filePath = path.join(tempDir, `audio_${timestamp}.mp3`);

    console.log(`[${timestamp}] Processing: ${videoUrl}`);

    try {
        if (!ytdl.validateURL(videoUrl)) {
            return res.status(400).json({ error: "Invalid YouTube URL" });
        }

        console.log("Downloading audio...");
        await new Promise((resolve, reject) => {
            const stream = ytdl(videoUrl, {
                quality: 'lowestaudio',
                filter: 'audioonly'
            });
            stream.pipe(fs.createWriteStream(filePath))
                .on('finish', resolve)
                .on('error', reject);
        });

        console.log("Uploading to Gemini...");
        const uploadResult = await fileManager.uploadFile(filePath, {
            mimeType: "audio/mp3",
            displayName: `Audio_${timestamp}`,
        });

        let file = await fileManager.getFile(uploadResult.file.name);
        while (file.state === "PROCESSING") {
            await new Promise((resolve) => setTimeout(resolve, 2000));
            file = await fileManager.getFile(uploadResult.file.name);
        }

        if (file.state === "FAILED") {
            throw new Error("Gemini failed to process the audio file.");
        }

        console.log("Generating transcript...");
        const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });

        const isOriginal = targetLanguage === 'Original';
        const prompt = `
            Transcribe this audio file exactly.
            ${isOriginal
                ? "Return the transcript in the original language spoken."
                : `Translate the transcript to ${targetLanguage}.`
            }
            Format: "[MM:SS] Speaker: Text"
            Do not include any other text, intro, or outro.
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
        console.error("Error:", error);
        if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
        res.status(500).json({ error: error.message || "Failed to transcribe" });
    }
});

app.listen(PORT, () => {
    console.log(`Backend running on port ${PORT}`);
});