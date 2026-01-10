# TubeScribe AI - System Manual

This document provides detailed instructions on how to set up, install, and run the TubeScribe AI system locally.

## 1. System Overview

TubeScribe AI consists of two main components:
*   **Frontend**: A React application (built with Vite) that provides the user interface.
*   **Backend**: A Node.js/Express server that handles YouTube transcript fetching and audio processing (fallback).

## 2. Prerequisites

Before you begin, ensure you have the following installed on your system:
*   **Node.js**: Version 20 or higher is recommended (LTS). Download from [nodejs.org](https://nodejs.org/).
*   **npm**: Included with Node.js.
*   **Git**: For cloning the repository.
*   **Gemini API Key**: You need an API key from Google AI Studio. Get it [here](https://aistudio.google.com/app/apikey).

## 3. Installation

The system has dependencies for both the frontend and the backend. You need to install them separately.

### 3.1. Frontend Installation

1.  Open your terminal/command prompt.
2.  Navigate to the project root directory (`tubescribe-ai`).
3.  Run the following command:
    ```bash
    npm install
    ```

### 3.2. Backend Installation

1.  Navigate to the `server` directory:
    ```bash
    cd server
    ```
2.  Run the following command:
    ```bash
    npm install
    ```

## 4. Configuration

You typically need to configure environment variables for both the frontend and the backend.

### 4.1. Frontend Configuration

1.  In the project root (`tubescribe-ai`), create a file named `.env.local` (if it doesn't exist).
2.  Add your Gemini API key to this file:
    ```env
    GEMINI_API_KEY=your_actual_api_key_here
    ```

### 4.2. Backend Configuration

1.  In the `server` directory, create a file named `.env`.
2.  Add your Gemini API key to this file as well (the backend uses it for translation):
    ```env
    GEMINI_API_KEY=your_actual_api_key_here
    ```

### 4.3. Linking Frontend to Local Backend

By default, the frontend is configured to point to a deployed backend on Hugging Face. To use your local backend:
1.  Open `services/geminiService.ts`.
2.  Locate the `BACKEND_URL` constant (around line 15).
3.  Change it to point to your local server:
    ```typescript
    // const BACKEND_URL = "https://kaiwen03-tubescribe-backend.hf.space";
    const BACKEND_URL = "http://localhost:7860/api/transcribe";
    ```
    *Note: Ensure the path `/api/transcribe` matches the route defined in `server/index.js`.*

## 5. Running the System

You need to run both the backend and frontend terminals simultaneously (or use a tool like `concurrently`, though running in separate terminals is simpler).

### 5.1. Start the Backend

1.  Open a terminal.
2.  Navigate to the `server` directory.
3.  Run the server:
    ```bash
    npm start
    ```
    You should see: `Backend running on port 7860`.

### 5.2. Start the Frontend

1.  Open a **new** terminal.
2.  Navigate to the project root (`tubescribe-ai`).
3.  Start the development server:
    ```bash
    npm run dev
    ```
4.  The terminal will show a local URL (usually `http://localhost:3000`). Open this link in your browser.

## 6. Troubleshooting

*   **"API_KEY environment variable is missing"**: Ensure you created `.env.local` in the root with `GEMINI_API_KEY`.
*   **Backend Connection Failed**: Ensure the backend server is running on port 7860 and you have updated `BACKEND_URL` in `geminiService.ts` correctly.
*   **"No transcript found"**: Some videos restrict captions. The system tries to fall back to the backend to download audio, but this requires the backend to be running and `ffmpeg` dependencies (handled by `youtube-transcript` or internal libs) to be working.
