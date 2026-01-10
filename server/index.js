// 1. Force IPv4 for DNS resolution (The "Node 20 Trick")
const dns = require('node:dns');
dns.setDefaultResultOrder('ipv4first');

const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');
require('dotenv').config();

// 2. BULLETPROOF IMPORT: Handles both ESM and CommonJS exports
const YoutubeTranscript = require('youtube-transcript').YoutubeTranscript;

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 7860;
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

app.get('/', (req, res) => {
    res.send('TubeScribe Backend Running (v10: Fixed Import)');
});

const getVideoId = (url) => {
    try {
        if (url.includes('youtu.be')) return url.split('/').pop().split('?')[0];
        const urlObj = new URL(url);
        return urlObj.searchParams.get('v') || urlObj.pathname.split('/').pop();
    } catch (e) { return null; }
};

app.post('/api/transcribe', async (req, res) => {
    const { videoUrl, targetLanguage } = req.body;
    const videoId = getVideoId(videoUrl);

    if (!videoId) return res.status(400).json({ error: "Invalid YouTube URL" });

    try {
        // 3. FETCH TRANSCRIPT
        // We call the class method directly
        const transcriptData = await YoutubeTranscript.fetchTranscript(videoId);

        if (!transcriptData || transcriptData.length === 0) {
            throw new Error("No transcript found for this video.");
        }

        let fullTranscript = transcriptData.map(segment => {
            const totalSeconds = Math.floor(segment.offset / 1000);
            const mm = Math.floor(totalSeconds / 60).toString().padStart(2, '0');
            const ss = (totalSeconds % 60).toString().padStart(2, '0');
            return `[${mm}:${ss}] ${segment.text}`;
        }).join('\n');

        // 4. TRANSLATE (Gemini)
        if (targetLanguage && targetLanguage !== 'Original') {
            const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
            const result = await model.generateContent(`
                Translate the following transcript to ${targetLanguage}.
                Keep [MM:SS] timestamps. Return only translation.
                
                ${fullTranscript.substring(0, 30000)}
            `);
            fullTranscript = result.response.text();
        }

        res.json({ transcript: fullTranscript });

    } catch (error) {
        console.error("❌ API Error:", error.message);
        res.status(500).json({ error: `Server error: ${error.message}` });
    }
});

app.listen(PORT, () => {
    console.log(`Backend running on port ${PORT}`);
    // Check if the library loaded correctly in logs
    console.log('Library Status:', typeof YoutubeTranscript.fetchTranscript === 'function' ? '✅ Ready' : '❌ Failed');
});