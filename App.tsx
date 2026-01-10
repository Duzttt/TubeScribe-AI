import React, { useState, useEffect } from 'react';
import {
  FileText,
  Languages,
  Sparkles,
  Video,
  ArrowRight,
  Globe,
  MessageCircle,
  Home,
  Mic,
  Ear,
  Moon,
  Sun
} from 'lucide-react';
import YouTubeInput from './components/YouTubeInput';
import ResultCard from './components/ResultCard';
import ChatInterface from './components/ChatInterface';
import Menu from './components/Menu';
import {
  ProcessingStatus,
  TargetLanguage,
  ChatMessage
} from './types';
import * as GeminiService from './services/geminiService';
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
  const [transcript, setTranscript] = useState<string>('');
  const [summary, setSummary] = useState<string>('');
  const [translation, setTranslation] = useState<string>('');
  const [targetLang, setTargetLang] = useState<TargetLanguage>(TargetLanguage.ORIGINAL);
  const [translationSource, setTranslationSource] = useState<'transcript' | 'summary'>('transcript');

  // Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatLoading, setChatLoading] = useState(false);

  // Track specific loading states
  const [loadingTranscript, setLoadingTranscript] = useState(false);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [loadingTranslation, setLoadingTranslation] = useState(false);

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
    setSummary('');
    setTranslation('');
    setChatMessages([]);
  };

  const handleUrlChange = (newUrl: string) => {
    setVideoUrl(newUrl);
  };

  const handleGenerateTranscript = async () => {
    if (!videoUrl) return;
    setLoadingTranscript(true);
    setStatus(ProcessingStatus.PROCESSING);
    setActiveTab('transcript');

    try {
      const result = await GeminiService.generateTranscript(videoUrl, targetLang);
      setTranscript(result);
      setStatus(ProcessingStatus.COMPLETED);
    } catch (error) {
      console.error(error);
      setStatus(ProcessingStatus.ERROR);
    } finally {
      setLoadingTranscript(false);
    }
  };

  const handleGenerateSummary = async () => {
    if (!videoUrl) return;
    setLoadingSummary(true);
    setStatus(ProcessingStatus.PROCESSING);
    setActiveTab('summary');

    try {
      const result = await GeminiService.generateSummary(videoUrl, transcript, targetLang);
      setSummary(result);
      setStatus(ProcessingStatus.COMPLETED);
    } catch (error) {
      console.error(error);
      setStatus(ProcessingStatus.ERROR);
    } finally {
      setLoadingSummary(false);
    }
  };

  const handleTranslate = async (source: 'transcript' | 'summary' = 'transcript') => {
    const sourceText = source === 'transcript' ? transcript : summary;

    if (!sourceText) {
      alert(`Please generate a ${source} first to translate.`);
      return;
    }

    setLoadingTranslation(true);
    setStatus(ProcessingStatus.PROCESSING);
    setActiveTab('translation');
    setTranslationSource(source);

    try {
      const result = await GeminiService.translateContent(sourceText, targetLang);
      setTranslation(result);
      setStatus(ProcessingStatus.COMPLETED);
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

            {/* Controls */}
            {videoUrl && (
              <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-5 sm:p-6 space-y-5 animate-fade-in-up delay-100 transition-colors flex-shrink-0">
                <h3 className="font-bold text-lg sm:text-xl text-gray-900 dark:text-white flex items-center gap-3">
                  <Sparkles className="text-yellow-500" size={24} />
                  AI Actions
                </h3>

                {/* Global Language Selector */}
                <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-xl border border-gray-200 dark:border-gray-600">
                  <label className="flex items-center gap-2 text-gray-700 dark:text-gray-300 font-bold text-xs sm:text-sm uppercase tracking-wider mb-3">
                    <Globe size={16} />
                    Output Language
                  </label>
                  <div className="relative">
                    <select
                      value={targetLang}
                      onChange={(e) => setTargetLang(e.target.value as TargetLanguage)}
                      className="w-full appearance-none pl-4 pr-10 py-3 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-xl text-base sm:text-lg focus:ring-2 focus:ring-red-500 focus:border-red-500 outline-none text-gray-900 dark:text-white cursor-pointer transition-shadow"
                    >
                      {Object.values(TargetLanguage).map((lang) => (
                        <option key={lang} value={lang}>
                          {lang}
                        </option>
                      ))}
                    </select>
                    <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none text-gray-500 dark:text-gray-400">
                      <ArrowRight size={18} className="rotate-90" />
                    </div>
                  </div>
                  {isTranslating ? (
                    <div className="mt-3 text-xs sm:text-sm text-blue-600 dark:text-blue-400 flex items-center gap-2 font-medium">
                      <Ear size={14} />
                      <span>Auto-detecting source language</span>
                    </div>
                  ) : (
                    <div className="mt-3 text-xs sm:text-sm text-gray-500 dark:text-gray-400 flex items-center gap-2">
                      <Ear size={14} />
                      <span>Detecting original language (No translation)</span>
                    </div>
                  )}
                </div>

                <div className="space-y-4">
                  {/* Primary Action: Transcribe */}
                  <button
                    onClick={handleGenerateTranscript}
                    disabled={loadingTranscript}
                    className="flex items-center justify-between w-full px-5 py-4 sm:py-5 bg-blue-50 dark:bg-blue-900/30 hover:bg-blue-100 dark:hover:bg-blue-900/50 text-blue-700 dark:text-blue-300 rounded-xl border border-blue-200 dark:border-blue-800 transition-all font-medium disabled:opacity-50 disabled:cursor-not-allowed group"
                  >
                    <span className="flex items-center gap-4 text-left">
                      <div className="w-10 h-10 rounded-full bg-blue-100 dark:bg-blue-800 flex items-center justify-center">
                        <Mic size={20} className="text-blue-600 dark:text-blue-300" />
                      </div>
                      <div>
                        <div className="font-bold text-base sm:text-lg">
                          {isTranslating ? "Auto-Detect & Translate" : "Generate Transcript"}
                        </div>
                        <div className="text-xs sm:text-sm opacity-85 font-normal">
                          {isTranslating ? `Detect source & text to ${targetLang}` : "Convert Speech to Text (Original)"}
                        </div>
                      </div>
                    </span>
                    {loadingTranscript ? (
                      <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin flex-shrink-0"></div>
                    ) : (
                      <ArrowRight size={20} className="opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0" />
                    )}
                  </button>

                  {/* Secondary Actions */}
                  <div className="grid grid-cols-2 gap-3 sm:gap-4">
                    <button
                      onClick={handleGenerateSummary}
                      disabled={loadingSummary || !transcript}
                      className="flex flex-col items-center justify-center gap-2 sm:gap-3 px-3 py-4 sm:py-6 bg-purple-50 dark:bg-purple-900/30 hover:bg-purple-100 dark:hover:bg-purple-900/50 text-purple-700 dark:text-purple-300 rounded-xl border border-purple-200 dark:border-purple-800 transition-all font-bold text-sm sm:text-base disabled:opacity-50 disabled:cursor-not-allowed h-full"
                      title={!transcript ? "Generate Transcript first" : "Summarize video content"}
                    >
                      <Sparkles size={24} />
                      Summarize
                      {loadingSummary && <div className="w-4 h-4 border-2 border-purple-500 border-t-transparent rounded-full animate-spin mt-1"></div>}
                    </button>

                    <button
                      onClick={() => handleTranslate('transcript')}
                      disabled={loadingTranslation || !transcript}
                      className="flex flex-col items-center justify-center gap-2 sm:gap-3 px-3 py-4 sm:py-6 bg-green-50 dark:bg-green-900/30 hover:bg-green-100 dark:hover:bg-green-900/50 text-green-700 dark:text-green-300 rounded-xl border border-green-200 dark:border-green-800 transition-all font-bold text-sm sm:text-base disabled:opacity-50 disabled:cursor-not-allowed h-full"
                      title={!transcript ? "Generate Transcript first" : "Re-translate the current text"}
                    >
                      <Languages size={24} />
                      Translate Text
                      {loadingTranslation && <div className="w-4 h-4 border-2 border-green-500 border-t-transparent rounded-full animate-spin mt-1"></div>}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Results - Fills Height */}
          <div className="lg:col-span-7 flex flex-col h-full overflow-hidden">
            {/* Tabs */}
            {videoUrl && (
              <div className="flex-shrink-0 mb-4 flex items-center space-x-2 bg-gray-200/50 dark:bg-gray-800 p-2 rounded-xl w-full sm:w-fit overflow-x-auto border border-gray-200 dark:border-gray-700 no-scrollbar">
                <button
                  onClick={() => setActiveTab('transcript')}
                  className={`px-4 sm:px-6 py-2 sm:py-3 rounded-lg text-sm sm:text-lg font-medium transition-all whitespace-nowrap ${activeTab === 'transcript'
                      ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
                      : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-200/50 dark:hover:bg-gray-700/50'
                    }`}
                >
                  Transcript
                </button>
                <button
                  onClick={() => setActiveTab('summary')}
                  className={`px-4 sm:px-6 py-2 sm:py-3 rounded-lg text-sm sm:text-lg font-medium transition-all whitespace-nowrap ${activeTab === 'summary'
                      ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
                      : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-200/50 dark:hover:bg-gray-700/50'
                    }`}
                >
                  Summary
                </button>
                <button
                  onClick={() => setActiveTab('translation')}
                  className={`px-4 sm:px-6 py-2 sm:py-3 rounded-lg text-sm sm:text-lg font-medium transition-all whitespace-nowrap ${activeTab === 'translation'
                      ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
                      : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-200/50 dark:hover:bg-gray-700/50'
                    }`}
                >
                  Translation
                </button>
                <button
                  onClick={() => setActiveTab('chat')}
                  className={`px-4 sm:px-6 py-2 sm:py-3 rounded-lg text-sm sm:text-lg font-medium transition-all whitespace-nowrap flex items-center gap-2 ${activeTab === 'chat'
                      ? 'bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm'
                      : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-200/50 dark:hover:bg-gray-700/50'
                    }`}
                >
                  <MessageCircle size={20} />
                  Chat
                </button>
              </div>
            )}

            {/* UPDATED: Container uses flex-grow and h-0 to allow children to handle scrolling */}
            <div className="flex-grow h-0 min-h-[500px] lg:min-h-0">
              {activeTab === 'transcript' && (
                <ResultCard
                  title={`Transcript ${isTranslating ? `(${targetLang})` : '(Original)'}`}
                  content={transcript}
                  isLoading={loadingTranscript}
                  type="markdown"
                  icon={<FileText size={24} className="text-blue-600 dark:text-blue-400" />}
                />
              )}
              {activeTab === 'summary' && (
                <ResultCard
                  title={`Summary ${isTranslating ? `(${targetLang})` : ''}`}
                  content={summary}
                  isLoading={loadingSummary}
                  type="markdown"
                  icon={<Sparkles size={24} className="text-purple-600 dark:text-purple-400" />}
                />
              )}
              {activeTab === 'translation' && (
                <ResultCard
                  title={`Translation (${targetLang}) - Source: ${translationSource}`}
                  content={translation}
                  isLoading={loadingTranslation}
                  type="markdown"
                  icon={<Languages size={24} className="text-green-600 dark:text-green-400" />}
                />
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