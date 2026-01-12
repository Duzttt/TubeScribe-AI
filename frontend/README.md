# Frontend

This directory contains the React frontend for TubeScribe AI.

## Setup

### Install Dependencies
```bash
npm install
```

### Environment Variables

Create a `.env.local` file in this directory:

```env
VITE_API_KEY=your_gemini_api_key_here
```

> **Note:** The Gemini API key is optional. You need it only for translation and chat features. Transcription and summarization work without it (uses local Python backend).

### Development

Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Production Build

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Project Structure

- `App.tsx` - Main application component
- `index.tsx` - Entry point
- `components/` - React components
- `services/` - API service modules
- `types.ts` - TypeScript type definitions
- `vite.config.ts` - Vite configuration
- `tailwind.config.js` - Tailwind CSS configuration
