const express = require('express');
const cors = require('cors');
const { Innertube, UniversalCache } = require('youtubei.js');
const { GoogleGenerativeAI } = require('@google/generative-ai');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 7860;
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

app.get('/', (req, res) => {
    res.send('TubeScribe Backend Running (v8: Innertube API)');
});

// Helper: Extract Video ID from any URL
const getVideoId = (url) => {
    try {
        if (url.includes('youtu.be')) return url.split('/').pop().split('?')[0];
        const urlObj = new URL(url);
        if (urlObj.hostname.includes('notegpt.io')) return urlObj.pathname.split('/').pop();
        return urlObj.searchParams.get('v');
    } catch (e) { return null; }
};

app.post('/api/transcribe', async (req, res) => {
    const { videoUrl, targetLanguage } = req.body;
    const videoId = getVideoId(videoUrl);

    console.log(`Processing ID: ${videoId}`);

    try {
        // 1. Initialize YouTube Client (Mimic Android App)
        const youtube = await Innertube.create({ cache: new UniversalCache(false), generate_session_locally: true });

        // 2. Fetch Video Info
        const info = await youtube.getInfo(videoId);

        // 3. Get Transcript Data
        const transcriptData = await info.getTranscript();

        if (!transcriptData || !transcriptData.transcript) {
            throw new Error("No transcript found for this video.");
        }

        // 4. Format Transcript
        // The API returns segments. We map them to "[MM:SS] Text"
        let fullTranscript = transcriptData.transcript.content.body.initial_segments.map(segment => {
            const startMs = parseInt(segment.start_ms);
            const totalSeconds = Math.floor(startMs / 1000);
            const mm = Math.floor(totalSeconds / 60).toString().padStart(2, '0');
            const ss = (totalSeconds % 60).toString().padStart(2, '0');
            const text = segment.snippet.text;
            return `[${mm}:${ss}] ${text}`;
        }).join('\n');

        console.log(`✅ Transcript fetched (${fullTranscript.length} chars)`);

        // 5. Translate if needed (Gemini)
        if (targetLanguage !== 'Original') {
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

        // Handle specific "No Caption" errors
        if (error.message.includes("No transcript")) {
            return res.status(404).json({ error: "This video does not have captions/transcript available." });
        }

        res.status(500).json({ error: `Failed to process: ${error.message}` });
    }
});

app.listen(PORT, () => {
    console.log(`Backend running on port ${PORT}`);
});