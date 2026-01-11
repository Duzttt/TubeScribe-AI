import React, { useState, useEffect } from 'react';
import {
  FileText,
  Languages,
  Sparkles,
  Video,
  ArrowRight,
  ArrowDown,
  Globe,
  MessageCircle,
  Home,
  Mic,
  Ear,
  Moon,
  Sun,
  Check,
  Circle
} from 'lucide-react';
import YouTubeInput from './components/YouTubeInput';
import ResultCard from './components/ResultCard';
import ChatInterface from './components/ChatInterface';
import Menu from './components/Menu';
import KeywordsDisplay from './components/KeywordsDisplay';
import ProgressBar from './components/ProgressBar';
import {
  ProcessingStatus,
  TargetLanguage,
  ChatMessage
} from './types';
import * as GeminiService from './services/geminiService';
import * as SummarizationService from './services/summarizationService';
import './index.css';

const App: React.FC = () => {
  const [view, setView] = useState<'menu' | 'app'>('menu');
  const [videoUrl, setVideoUrl] = useState('');

  // Theme State
  const [isDarkMode, setIsDarkMode] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('theme') === 'dark' ||
        (!localStorage.getItem('theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
    }
    return false;
  });

  const [status, setStatus] = useState<ProcessingStatus>(ProcessingStatus.IDLE);
  const [transcript, setTranscript] = useState<string>('');  // Always stores original transcript
  const [translatedTranscript, setTranslatedTranscript] = useState<string>(''); // Translated version for display
  const [detectedLanguage, setDetectedLanguage] = useState<string>(''); // Source language detected
  const [summary, setSummary] = useState<string>('');
  const [keywords, setKeywords] = useState<string[]>([]);
  const [targetLang, setTargetLang] = useState<TargetLanguage>(TargetLanguage.ORIGINAL);

  // Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatLoading, setChatLoading] = useState(false);

  // Track specific loading states
  const [loadingTranscript, setLoadingTranscript] = useState(false);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [loadingTranslation, setLoadingTranslation] = useState(false);
  
  // Progress tracking with smoothing
  const [transcriptionProgress, setTranscriptionProgress] = useState<{
    status: string;
    progress: number;
    message: string;
    downloadProgress?: number;
    transcriptionProgress?: number;
  } | null>(null);
  const [displayProgress, setDisplayProgress] = useState(0); // Smoothed progress for display

  const [activeTab, setActiveTab] = useState<'transcript' | 'summary' | 'translation' | 'chat'>('transcript');

  // Toggle Dark Mode
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDarkMode]);

  // Clear translated transcript when language changes (force re-translation)
  useEffect(() => {
    setTranslatedTranscript('');
    // Also clear summary since it depends on the translated transcript
    setSummary('');
    setKeywords([]);
  }, [targetLang]);

  // Smooth progress animation
  useEffect(() => {
    if (!transcriptionProgress) {
      setDisplayProgress(0);
      return;
    }

    const targetProgress = transcriptionProgress.progress || 0;
    
    // Animate progress smoothly
    const interval = setInterval(() => {
      setDisplayProgress((prev) => {
        const currentDiff = targetProgress - prev;
        // If difference is small, snap to target
        if (Math.abs(currentDiff) < 0.5) {
          clearInterval(interval);
          return targetProgress;
        }
        // Move 20% closer each frame (smooth animation)
        return prev + currentDiff * 0.2;
      });
    }, 50); // Update every 50ms for smooth animation

    return () => clearInterval(interval);
  }, [transcriptionProgress?.progress]);

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  const handleStart = () => {
    setView('app');
  };

  const handleReturnToMenu = () => {
    setView('menu');
  };

  const handleUrlSubmit = () => {
    setStatus(ProcessingStatus.IDLE);
    setTranscript('');
    setTranslatedTranscript('');
    setDetectedLanguage('');
    setSummary('');
    setKeywords([]);
    setChatMessages([]);
    setActiveTab('transcript');
  };

  const handleUrlChange = (newUrl: string) => {
    setVideoUrl(newUrl);
  };

  const handleGenerateTranscript = async () => {
    if (!videoUrl) return;
    setLoadingTranscript(true);
    setStatus(ProcessingStatus.PROCESSING);
    setActiveTab('transcript');
    setTranscriptionProgress({ status: 'starting', progress: 0, message: 'Initializing...' });
    setDisplayProgress(0);

    // Extract video ID for progress tracking
    const videoIdMatch = videoUrl.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})/);
    const videoId = videoIdMatch ? videoIdMatch[1] : null;

    // Set up Server-Sent Events for real-time progress updates
    let eventSource: EventSource | null = null;
    
    if (videoId) {
      try {
        eventSource = new EventSource(`http://localhost:8000/api/progress/${videoId}/stream`);
        
        eventSource.onmessage = (event) => {
          try {
            const progress = JSON.parse(event.data);
            setTranscriptionProgress(progress);
            
            // Progress smoothing is handled by useEffect
            if (progress.status === 'completed' || progress.status === 'error') {
              eventSource?.close();
              eventSource = null;
            }
          } catch (error) {
            console.warn('Failed to parse progress:', error);
          }
        };
        
        eventSource.onerror = (error) => {
          console.warn('SSE connection error:', error);
          eventSource?.close();
          eventSource = null;
        };
      } catch (error) {
        console.warn('Failed to connect to progress stream:', error);
        // Fallback to polling if SSE fails
        const progressInterval = setInterval(async () => {
          try {
              const response = await fetch(`http://localhost:8000/api/progress/${videoId}`);
            if (response.ok) {
              const progress = await response.json();
              setTranscriptionProgress(progress);
              
              // Progress smoothing is handled by useEffect
              if (progress.status === 'completed' || progress.status === 'error') {
                clearInterval(progressInterval);
              }
            }
          } catch (err) {
            console.warn('Failed to fetch progress:', err);
          }
        }, 1000);
        
        // Cleanup polling on completion
        setTimeout(() => clearInterval(progressInterval), 300000); // 5 min timeout
      }
    }

    try {
      // STEP 1: Always get ORIGINAL transcript (auto-detect language)
      console.log('📝 [FLOW] Step 1: Getting original transcript...');
      const originalResult = await GeminiService.generateTranscript(videoUrl, TargetLanguage.ORIGINAL);
      setTranscript(originalResult);  // Store original
      setDetectedLanguage('Auto-detected');
      
      // Don't auto-translate during transcription - let user do it in Step 2
      setTranslatedTranscript('');
      
      setStatus(ProcessingStatus.COMPLETED);
      setTranscriptionProgress({ status: 'completed', progress: 100, message: 'Completed!' });
      setDisplayProgress(100);
      
      // Stay on transcript tab to show result
      setActiveTab('transcript');
    } catch (error) {
      console.error(error);
      setStatus(ProcessingStatus.ERROR);
      setTranscriptionProgress({ status: 'error', progress: 0, message: 'Error occurred' });
      setDisplayProgress(0);
    } finally {
      if (eventSource) {
        eventSource.close();
      }
      setLoadingTranscript(false);
      // Clear progress after 3 seconds
      setTimeout(() => {
        setTranscriptionProgress(null);
        setDisplayProgress(0);
      }, 3000);
    }
  };

  const handleGenerateSummary = async () => {
    if (!videoUrl) return;
    
    // Require transcript for summarization
    if (!transcript || transcript.length < 50) {
      alert('Please generate a transcript first before summarizing.');
      return;
    }
    
    setLoadingSummary(true);
    setStatus(ProcessingStatus.PROCESSING);
    setActiveTab('summary');
    setKeywords([]); // Reset keywords

    try {
      // Check if Python backend is available
      const isBackendAvailable = await SummarizationService.checkBackendHealth();
      
      if (!isBackendAvailable) {
        throw new Error('Python backend is not available. Please make sure the backend server is running on port 8000.');
      }

      console.log('📊 [FLOW] Starting summarization flow...');
      
      // NEW FLOW: Transcript (original) → Translate → Summarize
      let textToSummarize = transcript; // Start with original transcript
      
      if (targetLang !== TargetLanguage.ORIGINAL) {
        // STEP 1: Translate transcript to target language first
        console.log(`📊 [FLOW] Step 1: Translating transcript to ${targetLang}...`);
        
        // Use already translated transcript if available, otherwise translate now
        if (translatedTranscript) {
          textToSummarize = translatedTranscript;
          console.log('📊 [FLOW] Using cached translated transcript');
        } else {
          console.log('📊 [FLOW] Translating transcript now...');
          textToSummarize = await GeminiService.translateContent(transcript, targetLang);
          setTranslatedTranscript(textToSummarize); // Cache it
        }
        
        // STEP 2: Summarize the TRANSLATED transcript
        // Since transcript is already in target language, summarize with AUTO (same language)
        console.log(`📊 [FLOW] Step 2: Summarizing translated transcript (already in ${targetLang})...`);
        const result = await SummarizationService.summarizeWithKeywords(
          textToSummarize,
          null // AUTO - summarize in the language of the input (which is already translated)
        );
        setSummary(result.summary);
        setKeywords(result.keywords);
        console.log(`✅ [FLOW] Summary generated in ${targetLang}`);
        setActiveTab('summary');
      } else {
        // Original language - just summarize directly
        console.log('📊 [FLOW] Summarizing original transcript (no translation)...');
        const result = await SummarizationService.summarizeWithKeywords(
          transcript,
          null // AUTO - keep in original language
        );
        setSummary(result.summary);
        setKeywords(result.keywords);
        console.log('✅ [FLOW] Summary generated in original language');
        setActiveTab('summary');
      }
      
      setStatus(ProcessingStatus.COMPLETED);
    } catch (error) {
      console.error(error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to generate summary. Please ensure the Python backend is running.';
      alert(errorMessage);
      setStatus(ProcessingStatus.ERROR);
    } finally {
      setLoadingSummary(false);
    }
  };

  const handleTranslate = async () => {
    if (!transcript) {
      alert('Please generate a transcript first to translate.');
      return;
    }

    if (!isTranslating) {
      alert('Please select a language other than "Original" to translate.');
      return;
    }

    setLoadingTranslation(true);
    setStatus(ProcessingStatus.PROCESSING);
    setActiveTab('translation');

    try {
      console.log(`🌍 [FLOW] Step 2: Translating transcript to ${targetLang}...`);
      const result = await GeminiService.translateContent(transcript, targetLang);
      setTranslatedTranscript(result);
      setStatus(ProcessingStatus.COMPLETED);
      console.log('✅ [FLOW] Translation complete');
    } catch (error) {
      console.error(error);
      setStatus(ProcessingStatus.ERROR);
    } finally {
      setLoadingTranslation(false);
    }
  };

  const handleSendMessage = async (text: string) => {
    if (!text.trim()) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      text,
      timestamp: Date.now()
    };

    setChatMessages(prev => [...prev, userMsg]);
    setChatLoading(true);

    try {
      const context = summary || transcript || '';
      const responseText = await GeminiService.sendChatMessage(chatMessages, text, videoUrl, context);

      const botMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'model',
        text: responseText,
        timestamp: Date.now()
      };

      setChatMessages(prev => [...prev, botMsg]);
    } catch (error) {
      console.error("Chat error:", error);
    } finally {
      setChatLoading(false);
    }
  };

  if (view === 'menu') {
    return <Menu onStart={handleStart} isDarkMode={isDarkMode} toggleTheme={toggleTheme} />;
  }

  const isTranslating = targetLang !== TargetLanguage.ORIGINAL;

  return (
    // UPDATED: Use lg:h-screen and lg:overflow-hidden to prevent body scroll on desktop
    <div className="min-h-screen lg:h-screen bg-gray-50 dark:bg-gray-900 flex flex-col transition-colors duration-300 overflow-y-auto lg:overflow-hidden">
      {/* Header */}
      <header className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-200 dark:border-gray-700 sticky top-0 z-20 flex-shrink-0">
        <div className="max-w-[1920px] mx-auto px-4 sm:px-6 lg:px-8 h-16 sm:h-20 flex items-center justify-between">
          <div className="flex items-center gap-3 sm:gap-5">
            <button
              onClick={handleReturnToMenu}
              className="p-2 sm:p-2.5 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full text-gray-500 dark:text-gray-400 transition-colors"
              title="Back to Menu"
            >
              <Home size={20} className="sm:w-6 sm:h-6" />
            </button>
            <div className="flex items-center gap-2 sm:gap-3">
              <div className="w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-br from-red-600 to-red-500 rounded-xl flex items-center justify-center text-white shadow-lg shadow-red-500/30">
                <Video size={18} className="sm:w-6 sm:h-6" fill="currentColor" />
              </div>
              <h1 className="text-lg sm:text-2xl font-bold text-gray-900 dark:text-white tracking-tight">TubeScribe <span className="text-red-600 dark:text-red-500">AI</span></h1>
            </div>
          </div>
          <div className="flex items-center gap-3 sm:gap-5">
            <button
              onClick={toggleTheme}
              className="p-2 sm:p-3 rounded-full text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800 transition-colors"
              title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
            >
              {isDarkMode ? <Sun size={20} className="sm:w-6 sm:h-6" /> : <Moon size={20} className="sm:w-6 sm:h-6" />}
            </button>
            <div className="hidden sm:flex items-center gap-6 text-base font-medium text-gray-500 dark:text-gray-400">
              <span className="text-xs sm:text-sm px-3 py-1.5 bg-gray-100 dark:bg-gray-800 rounded-md border border-gray-200 dark:border-gray-700">YouTube Mode</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow w-full max-w-[1920px] mx-auto px-4 sm:px-6 lg:px-8 py-4 lg:py-6 overflow-hidden">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-full">

          {/* Left Column: Input & Controls - SCROLLABLE on Desktop */}
          <div className="lg:col-span-5 flex flex-col gap-4 lg:h-full lg:overflow-y-auto lg:pr-2 custom-scrollbar">

            {/* YouTube Input Area */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-4 sm:p-5 transition-colors flex-shrink-0">
              <YouTubeInput
                url={videoUrl}
                onUrlChange={handleUrlChange}
                onSubmit={handleUrlSubmit}
                isLoading={status === ProcessingStatus.PROCESSING}
              />
            </div>

            {/* Controls - Step by Step Flow */}
            {videoUrl && (
              <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-5 sm:p-6 space-y-4 animate-fade-in-up delay-100 transition-colors flex-shrink-0">
                
                {/* Header with Language Selector */}
                <div className="flex items-center justify-between gap-4 pb-4 border-b border-gray-200 dark:border-gray-700">
                  <h3 className="font-bold text-lg sm:text-xl text-gray-900 dark:text-white flex items-center gap-2">
                    <Sparkles className="text-yellow-500" size={22} />
                    Processing Flow
                  </h3>
                  <div className="flex items-center gap-2">
                    <Globe size={16} className="text-gray-500 dark:text-gray-400" />
                    <select
                      value={targetLang}
                      onChange={(e) => setTargetLang(e.target.value as TargetLanguage)}
                      className="appearance-none pl-3 pr-8 py-2 bg-gray-100 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg text-sm focus:ring-2 focus:ring-red-500 focus:border-red-500 outline-none text-gray-900 dark:text-white cursor-pointer"
                    >
                      {Object.values(TargetLanguage).map((lang) => (
                        <option key={lang} value={lang}>
                          {lang}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Step-by-Step Flow */}
                <div className="space-y-3">
                  
                  {/* STEP 1: Transcribe */}
                  <div className="relative">
                    <button
                      onClick={handleGenerateTranscript}
                      disabled={loadingTranscript}
                      className={`flex items-center w-full px-4 py-4 rounded-xl border-2 transition-all font-medium disabled:cursor-not-allowed group
                        ${transcript 
                          ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300' 
                          : 'bg-blue-50 dark:bg-blue-900/30 hover:bg-blue-100 dark:hover:bg-blue-900/50 border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300'
                        }
                        ${loadingTranscript ? 'opacity-70' : ''}
                      `}
                    >
                      {/* Step Number */}
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center mr-4 flex-shrink-0 font-bold text-sm
                        ${transcript 
                          ? 'bg-blue-500 text-white' 
                          : 'bg-blue-200 dark:bg-blue-800 text-blue-700 dark:text-blue-300'
                        }`}
                      >
                        {transcript ? <Check size={16} /> : '1'}
                      </div>
                      
                      <div className="flex-grow text-left">
                        <div className="font-bold text-base flex items-center gap-2">
                          <Mic size={18} />
                          Transcribe
                          {transcript && <span className="text-xs font-normal text-blue-500 dark:text-blue-400">(Done)</span>}
                        </div>
                        <div className="text-xs opacity-75">
                          Convert speech to text (auto-detect language)
                        </div>
                      </div>
                      
                      {loadingTranscript ? (
                        <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin flex-shrink-0"></div>
                      ) : (
                        <ArrowRight size={18} className="opacity-50 group-hover:opacity-100 transition-opacity flex-shrink-0" />
                      )}
                    </button>
                  </div>

                  {/* Arrow Down */}
                  <div className="flex justify-center">
                    <ArrowDown size={20} className={`${transcript ? 'text-green-500' : 'text-gray-300 dark:text-gray-600'}`} />
                  </div>

                  {/* STEP 2: Translate (conditional) */}
                  <div className="relative">
                    <button
                      onClick={handleTranslate}
                      disabled={loadingTranslation || !transcript || !isTranslating}
                      className={`flex items-center w-full px-4 py-4 rounded-xl border-2 transition-all font-medium group
                        ${!isTranslating 
                          ? 'bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-400 dark:text-gray-500 cursor-not-allowed' 
                          : translatedTranscript
                            ? 'bg-green-50 dark:bg-green-900/20 border-green-300 dark:border-green-700 text-green-700 dark:text-green-300'
                            : !transcript
                              ? 'bg-gray-100 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-400 dark:text-gray-500 cursor-not-allowed'
                              : 'bg-green-50 dark:bg-green-900/30 hover:bg-green-100 dark:hover:bg-green-900/50 border-green-200 dark:border-green-800 text-green-700 dark:text-green-300'
                        }
                        ${loadingTranslation ? 'opacity-70' : ''}
                      `}
                    >
                      {/* Step Number */}
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center mr-4 flex-shrink-0 font-bold text-sm
                        ${!isTranslating 
                          ? 'bg-gray-200 dark:bg-gray-700 text-gray-400 dark:text-gray-500'
                          : translatedTranscript 
                            ? 'bg-green-500 text-white' 
                            : 'bg-green-200 dark:bg-green-800 text-green-700 dark:text-green-300'
                        }`}
                      >
                        {translatedTranscript && isTranslating ? <Check size={16} /> : '2'}
                      </div>
                      
                      <div className="flex-grow text-left">
                        <div className="font-bold text-base flex items-center gap-2">
                          <Languages size={18} />
                          Translate
                          {!isTranslating && <span className="text-xs font-normal">(Skipped - Original)</span>}
                          {translatedTranscript && isTranslating && <span className="text-xs font-normal text-green-500 dark:text-green-400">(Done)</span>}
                        </div>
                        <div className="text-xs opacity-75">
                          {isTranslating ? `Translate transcript to ${targetLang}` : 'Select a language to enable'}
                        </div>
                      </div>
                      
                      {loadingTranslation ? (
                        <div className="w-5 h-5 border-2 border-green-500 border-t-transparent rounded-full animate-spin flex-shrink-0"></div>
                      ) : isTranslating && transcript && !translatedTranscript ? (
                        <ArrowRight size={18} className="opacity-50 group-hover:opacity-100 transition-opacity flex-shrink-0" />
                      ) : null}
                    </button>
                    
                    {/* Auto-translate hint */}
                    {isTranslating && transcript && !translatedTranscript && (
                      <div className="mt-2 text-xs text-green-600 dark:text-green-400 flex items-center gap-1 pl-12">
                        <Ear size={12} />
                        Translation will run automatically with Summarize
                      </div>
                    )}
                  </div>

                  {/* Arrow Down */}
                  <div className="flex justify-center">
                    <ArrowDown size={20} className={`${(transcript && (!isTranslating || translatedTranscript)) ? 'text-purple-500' : 'text-gray-300 dark:text-gray-600'}`} />
                  </div>

                  {/* STEP 3: Summarize */}
                  <div className="relative">
                    <button
                      onClick={handleGenerateSummary}
                      disabled={loadingSummary || !transcript}
                      className={`flex items-center w-full px-4 py-4 rounded-xl border-2 transition-all font-medium group
                        ${summary 
                          ? 'bg-purple-50 dark:bg-purple-900/20 border-purple-300 dark:border-purple-700 text-purple-700 dark:text-purple-300' 
                          : !transcript
                            ? 'bg-gray-100 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-400 dark:text-gray-500 cursor-not-allowed'
                            : 'bg-purple-50 dark:bg-purple-900/30 hover:bg-purple-100 dark:hover:bg-purple-900/50 border-purple-200 dark:border-purple-800 text-purple-700 dark:text-purple-300'
                        }
                        ${loadingSummary ? 'opacity-70' : ''}
                      `}
                    >
                      {/* Step Number */}
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center mr-4 flex-shrink-0 font-bold text-sm
                        ${summary 
                          ? 'bg-purple-500 text-white' 
                          : !transcript
                            ? 'bg-gray-200 dark:bg-gray-700 text-gray-400 dark:text-gray-500'
                            : 'bg-purple-200 dark:bg-purple-800 text-purple-700 dark:text-purple-300'
                        }`}
                      >
                        {summary ? <Check size={16} /> : '3'}
                      </div>
                      
                      <div className="flex-grow text-left">
                        <div className="font-bold text-base flex items-center gap-2">
                          <Sparkles size={18} />
                          Summarize
                          {summary && <span className="text-xs font-normal text-purple-500 dark:text-purple-400">(Done)</span>}
                        </div>
                        <div className="text-xs opacity-75">
                          {isTranslating 
                            ? `Generate summary in ${targetLang}` 
                            : 'Generate summary in original language'
                          }
                        </div>
                      </div>
                      
                      {loadingSummary ? (
                        <div className="w-5 h-5 border-2 border-purple-500 border-t-transparent rounded-full animate-spin flex-shrink-0"></div>
                      ) : transcript ? (
                        <ArrowRight size={18} className="opacity-50 group-hover:opacity-100 transition-opacity flex-shrink-0" />
                      ) : null}
                    </button>
                  </div>
                </div>

                {/* Flow Summary */}
                <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                  <div className="text-xs text-gray-500 dark:text-gray-400 flex items-center justify-center gap-2">
                    <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded">Transcribe</span>
                    <ArrowRight size={14} />
                    {isTranslating ? (
                      <>
                        <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded">Translate</span>
                        <ArrowRight size={14} />
                      </>
                    ) : (
                      <>
                        <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-400 dark:text-gray-500 rounded line-through">Translate</span>
                        <ArrowRight size={14} className="text-gray-300" />
                      </>
                    )}
                    <span className="px-2 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded">Summarize</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Results - Fills Height */}
          <div className="lg:col-span-7 flex flex-col h-full overflow-hidden">
            {/* Tabs - Simplified to match flow */}
            {videoUrl && (
              <div className="flex-shrink-0 mb-4 flex items-center space-x-2 bg-gray-200/50 dark:bg-gray-800 p-2 rounded-xl w-full sm:w-fit overflow-x-auto border border-gray-200 dark:border-gray-700 no-scrollbar">
                {/* Step 1: Transcript */}
                <button
                  onClick={() => setActiveTab('transcript')}
                  className={`px-4 sm:px-6 py-2 sm:py-3 rounded-lg text-sm sm:text-lg font-medium transition-all whitespace-nowrap flex items-center gap-2 ${activeTab === 'transcript'
                      ? 'bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm'
                      : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-200/50 dark:hover:bg-gray-700/50'
                    }`}
                >
                  <span className={`w-5 h-5 rounded-full text-xs flex items-center justify-center ${
                    transcript ? 'bg-blue-500 text-white' : 'bg-gray-300 dark:bg-gray-600 text-gray-600 dark:text-gray-400'
                  }`}>
                    {transcript ? <Check size={12} /> : '1'}
                  </span>
                  Transcript
                </button>
                
                {/* Arrow */}
                <ArrowRight size={16} className="text-gray-400 dark:text-gray-500 flex-shrink-0" />
                
                {/* Step 2: Translated (only show if translating) */}
                {isTranslating && (
                  <>
                    <button
                      onClick={() => setActiveTab('translation')}
                      className={`px-4 sm:px-6 py-2 sm:py-3 rounded-lg text-sm sm:text-lg font-medium transition-all whitespace-nowrap flex items-center gap-2 ${activeTab === 'translation'
                          ? 'bg-white dark:bg-gray-700 text-green-600 dark:text-green-400 shadow-sm'
                          : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-200/50 dark:hover:bg-gray-700/50'
                        }`}
                    >
                      <span className={`w-5 h-5 rounded-full text-xs flex items-center justify-center ${
                        translatedTranscript ? 'bg-green-500 text-white' : 'bg-gray-300 dark:bg-gray-600 text-gray-600 dark:text-gray-400'
                      }`}>
                        {translatedTranscript ? <Check size={12} /> : '2'}
                      </span>
                      Translated
                    </button>
                    <ArrowRight size={16} className="text-gray-400 dark:text-gray-500 flex-shrink-0" />
                  </>
                )}
                
                {/* Step 3: Summary */}
                <button
                  onClick={() => setActiveTab('summary')}
                  className={`px-4 sm:px-6 py-2 sm:py-3 rounded-lg text-sm sm:text-lg font-medium transition-all whitespace-nowrap flex items-center gap-2 ${activeTab === 'summary'
                      ? 'bg-white dark:bg-gray-700 text-purple-600 dark:text-purple-400 shadow-sm'
                      : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-200/50 dark:hover:bg-gray-700/50'
                    }`}
                >
                  <span className={`w-5 h-5 rounded-full text-xs flex items-center justify-center ${
                    summary ? 'bg-purple-500 text-white' : 'bg-gray-300 dark:bg-gray-600 text-gray-600 dark:text-gray-400'
                  }`}>
                    {summary ? <Check size={12} /> : isTranslating ? '3' : '2'}
                  </span>
                  Summary
                </button>
                
                {/* Divider */}
                <div className="w-px h-6 bg-gray-300 dark:bg-gray-600 mx-1"></div>
                
                {/* Chat */}
                <button
                  onClick={() => setActiveTab('chat')}
                  className={`px-4 sm:px-6 py-2 sm:py-3 rounded-lg text-sm sm:text-lg font-medium transition-all whitespace-nowrap flex items-center gap-2 ${activeTab === 'chat'
                      ? 'bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm'
                      : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-200/50 dark:hover:bg-gray-700/50'
                    }`}
                >
                  <MessageCircle size={18} />
                  Chat
                </button>
              </div>
            )}

            {/* UPDATED: Container uses flex-grow and h-0 to allow children to handle scrolling */}
            <div className="flex-grow h-0 min-h-[500px] lg:min-h-0">
              {activeTab === 'transcript' && (
                <div className="flex flex-col h-full gap-4">
                  {transcriptionProgress && loadingTranscript && (
                    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 flex-shrink-0">
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <h3 className="font-bold text-lg text-gray-800 dark:text-gray-200">
                            {transcriptionProgress.status === 'downloading' && '⬇️ Downloading Audio'}
                            {transcriptionProgress.status === 'transcribing' && '🎤 Processing Audio'}
                            {transcriptionProgress.status === 'completed' && '✅ Completed'}
                            {transcriptionProgress.status === 'error' && '❌ Error'}
                          </h3>
                          <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                            {Math.round(displayProgress)}%
                          </span>
                        </div>
                        
                        <ProgressBar
                          progress={displayProgress}
                          label={transcriptionProgress.message || 'Processing...'}
                          color={
                            transcriptionProgress.status === 'downloading' ? 'blue' :
                            transcriptionProgress.status === 'transcribing' ? 'purple' :
                            transcriptionProgress.status === 'completed' ? 'green' : 'blue'
                          }
                        />
                      </div>
                    </div>
                  )}
                  
                  <div className="flex-grow min-h-0">
                    <ResultCard
                      title="Step 1: Transcript (Original)"
                      content={transcript}
                      isLoading={loadingTranscript}
                      type="markdown"
                      icon={<FileText size={24} className="text-blue-600 dark:text-blue-400" />}
                    />
                  </div>
                </div>
              )}
              {activeTab === 'translation' && (
                <div className="flex flex-col h-full gap-4">
                  <div className="flex-grow min-h-0">
                    <ResultCard
                      title={`Step 2: Translated Transcript (${targetLang})`}
                      content={translatedTranscript || (transcript ? 'Click "Translate" in the Processing Flow to translate the transcript.' : 'Generate a transcript first.')}
                      isLoading={loadingTranslation}
                      type="markdown"
                      icon={<Languages size={24} className="text-green-600 dark:text-green-400" />}
                    />
                  </div>
                </div>
              )}
              {activeTab === 'summary' && (
                <div className="flex flex-col h-full gap-4">
                  {keywords.length > 0 && (
                    <KeywordsDisplay keywords={keywords} className="flex-shrink-0" />
                  )}
                  <div className="flex-grow min-h-0">
                    <ResultCard
                      title={`Step ${isTranslating ? '3' : '2'}: Summary ${isTranslating ? `(${targetLang})` : '(Original)'}`}
                      content={summary || (transcript ? 'Click "Summarize" in the Processing Flow to generate a summary.' : 'Generate a transcript first.')}
                      isLoading={loadingSummary}
                      type="markdown"
                      icon={<Sparkles size={24} className="text-purple-600 dark:text-purple-400" />}
                    />
                  </div>
                </div>
              )}
              {activeTab === 'chat' && (
                <ChatInterface
                  messages={chatMessages}
                  onSendMessage={handleSendMessage}
                  isLoading={chatLoading}
                />
              )}

              {!videoUrl && (
                <div className="h-full flex flex-col items-center justify-center text-gray-400 dark:text-gray-500 p-8 sm:p-12 border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-2xl bg-white/50 dark:bg-gray-800/50 transition-colors">
                  <div className="w-16 h-16 sm:w-24 sm:h-24 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center mb-6 shadow-inner">
                    <ArrowRight size={32} className="sm:w-10 sm:h-10 text-gray-300 dark:text-gray-600" />
                  </div>
                  <p className="text-lg sm:text-2xl font-medium text-center">Paste a YouTube URL to get started</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;