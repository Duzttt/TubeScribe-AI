// 1. THE "NODE 20 TRICK": Force IPv4 to prevent DNS resolution errors
const dns = require('node:dns');
dns.setDefaultResultOrder('ipv4first');

const express = require('express');
const cors = require('cors');
const { YouTubeTranscript } = require('youtube-transcript'); // Lighter, more reliable
const { GoogleGenerativeAI } = require('@google/generative-ai');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 7860;
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

app.get('/', (req, res) => {
    res.send('TubeScribe Backend Running (v9: Node 20 + YouTubeTranscript)');
});

// Helper: Extract Video ID
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

    if (!videoId) {
        return res.status(400).json({ error: "Invalid YouTube URL" });
    }

    console.log(`Processing ID: ${videoId}`);

    try {
        // 2. FETCH TRANSCRIPT
        // This library fetches the web-accessible XML captions
        const transcriptData = await YouTubeTranscript.fetchTranscript(videoId);

        if (!transcriptData || transcriptData.length === 0) {
            throw new Error("No transcript found for this video.");
        }

        // 3. FORMAT TRANSCRIPT with [MM:SS]
        let fullTranscript = transcriptData.map(segment => {
            const totalSeconds = Math.floor(segment.offset / 1000);
            const mm = Math.floor(totalSeconds / 60).toString().padStart(2, '0');
            const ss = (totalSeconds % 60).toString().padStart(2, '0');
            return `[${mm}:${ss}] ${segment.text}`;
        }).join('\n');

        console.log(`✅ Transcript fetched (${fullTranscript.length} chars)`);

        // 4. TRANSLATE (Gemini)
        if (targetLanguage && targetLanguage !== 'Original') {
            console.log(`Translating to ${targetLanguage}...`);
            const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
            const result = await model.generateContent(`
                Translate the following transcript to ${targetLanguage}.
                IMPORTANT: Keep the [MM:SS] timestamps at the start of lines.
                Do not summarize. Return only the translated transcript.
                
                TRANSCRIPT START:
                ${fullTranscript.substring(0, 30000)}
            `);
            fullTranscript = result.response.text();
        }

        res.json({ transcript: fullTranscript });

    } catch (error) {
        console.error("❌ Transcription Error:", error);

        if (error.message.includes("transcript is disabled") || error.message.includes("No transcript")) {
            return res.status(404).json({ error: "Captions are disabled or unavailable for this video." });
        }

        res.status(500).json({ error: `Server error: ${error.message}` });
    }
});

app.listen(PORT, () => {
    console.log(`Backend running on port ${PORT}`);
});