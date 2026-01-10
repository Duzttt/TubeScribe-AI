import React, { useState, useRef, useEffect } from 'react';
import { Video, ArrowRight, Sparkles, Send, Bot, Sun, Moon } from 'lucide-react';
import { ChatMessage } from '../types';
import * as GeminiService from '../services/geminiService';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface MenuProps {
  onStart: () => void;
  isDarkMode?: boolean;
  toggleTheme?: () => void;
}

const Menu: React.FC<MenuProps> = ({ onStart, isDarkMode, toggleTheme }) => {
  const [guideMessages, setGuideMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'model',
      text: "Hi! I'm the TubeScribe Guide. \n\nI can explain the project, features, or help you get started.\n\nTry asking:\n* \"What can this app do?\"\n* \"How do I translate a video?\"\n* \"Take me to the app!\"",
      timestamp: Date.now()
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [guideMessages]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      text: input,
      timestamp: Date.now()
    };

    setGuideMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      let responseText = await GeminiService.sendGuideMessage(guideMessages, userMsg.text);

      if (responseText.includes('[ACTION:START_APP]')) {
        responseText = responseText.replace('[ACTION:START_APP]', '');
        setTimeout(() => {
          onStart();
        }, 1500);
      }

      setGuideMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        role: 'model',
        text: responseText,
        timestamp: Date.now()
      }]);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    // Main Container: Full Screen, No Scroll on Body
    <div className="min-h-screen lg:h-screen bg-gray-50 dark:bg-gray-950 flex flex-col transition-colors duration-500 relative overflow-x-hidden lg:overflow-hidden">

      {/* Custom CSS for Animations */}
      <style>{`
        @keyframes gradient-shift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        @keyframes liquid-shimmer {
          0% { background-position: 0% 50%; }
          100% { background-position: 200% 50%; }
        }
        @keyframes glow-pulse {
          0%, 100% { filter: drop-shadow(0 0 2px rgba(255, 255, 255, 0.6)); transform: scale(1); }
          50% { filter: drop-shadow(0 0 10px rgba(255, 165, 0, 0.8)); transform: scale(1.05); }
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-10px); }
        }
        .animate-gradient-shift {
          background-size: 200% 200%;
          animation: gradient-shift 3s ease infinite;
        }
        .animate-liquid {
          background-size: 200% auto;
          animation: liquid-shimmer 3s linear infinite;
        }
        .animate-icon-glow {
          animation: glow-pulse 3s ease-in-out infinite;
        }
        .animate-float {
          animation: float 4s ease-in-out infinite;
        }
        .animate-float-delay-1 {
          animation: float 4s ease-in-out infinite;
          animation-delay: 1s;
        }
        .animate-float-delay-2 {
          animation: float 4s ease-in-out infinite;
          animation-delay: 2s;
        }
      `}</style>

      {/* Background Decor */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[600px] bg-red-500/10 dark:bg-red-900/10 rounded-[100%] blur-3xl -z-10 pointer-events-none" />

      {/* Nav */}
      <nav className="p-4 flex justify-end flex-shrink-0 z-10">
        {toggleTheme && (
          <button
            onClick={toggleTheme}
            className="p-2 sm:p-2.5 rounded-full bg-white dark:bg-gray-800 shadow-sm border border-gray-100 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-all"
          >
            {isDarkMode ? <Sun size={18} className="sm:w-5 sm:h-5" /> : <Moon size={18} className="sm:w-5 sm:h-5" />}
          </button>
        )}
      </nav>

      {/* Main Content Area */}
      <div className="flex-grow flex flex-col p-4 lg:p-8 overflow-hidden">
        {/* Grid: Left=5, Right=3 (3/8ths) */}
        <div className="max-w-[1920px] w-full mx-auto grid grid-cols-1 lg:grid-cols-8 gap-6 lg:gap-12 flex-grow h-full text-center lg:text-left">

          {/* Left Column: Intro & Actions (Takes 5/8 space) */}
          <div className="lg:col-span-5 flex flex-col justify-center space-y-8 animate-fade-in-up py-4 lg:py-0 overflow-y-auto custom-scrollbar h-full">
            <div className="space-y-6">

              {/* ANIMATED LOGO */}
              <div className="mx-auto lg:mx-0 w-20 h-20 sm:w-24 sm:h-24 bg-gradient-to-br from-red-600 via-orange-500 to-yellow-500 animate-gradient-shift rounded-3xl shadow-2xl flex items-center justify-center text-white transform hover:scale-105 transition-transform duration-300 ring-4 ring-red-100 dark:ring-red-900/30">
                <Video size={40} className="sm:w-12 sm:h-12 animate-icon-glow" fill="currentColor" />
              </div>

              <div>
                <h1 className="text-4xl sm:text-5xl lg:text-7xl font-black text-gray-900 dark:text-white tracking-tight leading-[1.1]">
                  TubeScribe
                  {/* ANIMATED "AI" TEXT */}
                  <span className="ml-2 text-transparent bg-clip-text bg-gradient-to-r from-red-600 via-purple-500 to-pink-600 animate-liquid">
                    AI
                  </span>
                </h1>
                <p className="mt-6 text-lg sm:text-xl text-gray-600 dark:text-gray-300 max-w-lg mx-auto lg:mx-0 leading-relaxed font-light">
                  Unlimited, batch transcribe YouTube videos and playlists into accurate text in any language in just seconds.
                  <span className="block mt-2 font-normal text-gray-900 dark:text-gray-100">Transcribe. Summarize. Translate. Chat.</span>
                </p>
              </div>
            </div>

            {/* ANIMATED FEATURE ICONS (Floating) */}
            <div className="grid grid-cols-3 gap-3 sm:gap-4 max-w-lg mx-auto lg:mx-0 w-full">
              {/* Icon 1: Transcribe */}
              <div className="animate-float flex flex-col items-center text-center p-3 sm:p-4 bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 transition-colors">
                <div className="p-2 sm:p-3 bg-purple-50 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 rounded-xl mb-2"><span className="font-bold text-xl leading-none">Aa</span></div>
                <span className="font-bold text-gray-800 dark:text-gray-200 text-xs sm:text-sm">Transcribe</span>
              </div>

              {/* Icon 2: Translate (Delayed) */}
              <div className="animate-float-delay-1 flex flex-col items-center text-center p-3 sm:p-4 bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 transition-colors">
                <div className="p-2 sm:p-3 bg-green-50 dark:bg-green-900/30 text-green-600 dark:text-green-400 rounded-xl mb-2"><span className="font-bold text-xl leading-none">文</span></div>
                <span className="font-bold text-gray-800 dark:text-gray-200 text-xs sm:text-sm">Translate</span>
              </div>

              {/* Icon 3: Summarize (Delayed) */}
              <div className="animate-float-delay-2 flex flex-col items-center text-center p-3 sm:p-4 bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 transition-colors">
                <div className="p-2 sm:p-3 bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-xl mb-2"><Sparkles size={20} className="sm:w-6 sm:h-6" /></div>
                <span className="font-bold text-gray-800 dark:text-gray-200 text-xs sm:text-sm">Summarize</span>
              </div>
            </div>

            <button
              onClick={onStart}
              className="w-full sm:w-auto py-4 px-10 bg-gray-900 dark:bg-white hover:bg-black dark:hover:bg-gray-200 text-white dark:text-gray-900 rounded-2xl font-bold text-lg shadow-xl shadow-gray-200 dark:shadow-none flex items-center justify-center gap-4 group transition-all transform hover:-translate-y-1 mx-auto lg:mx-0"
            >
              Start Analyzing
              <ArrowRight size={24} className="group-hover:translate-x-1 transition-transform" />
            </button>
          </div>

          {/* Right Column: Embedded Chat Guide (Takes 3/8 space) */}
          <div className="lg:col-span-3 w-full h-[500px] lg:h-full max-h-[85vh] bg-white dark:bg-gray-900/80 rounded-[2rem] shadow-2xl border border-gray-200 dark:border-gray-800 overflow-hidden flex flex-col animate-fade-in-up delay-100 relative ring-8 ring-gray-100/50 dark:ring-gray-800/30 backdrop-blur-sm mb-6 lg:mb-0">
            {/* Chat Header */}
            <div className="bg-white/90 dark:bg-gray-900/90 backdrop-blur border-b border-gray-100 dark:border-gray-800 p-4 sm:p-5 flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-yellow-400 to-orange-400 rounded-2xl flex items-center justify-center text-white shadow-lg shadow-orange-200 dark:shadow-none flex-shrink-0">
                <Bot size={20} />
              </div>
              <div>
                <h3 className="font-bold text-gray-900 dark:text-white text-base sm:text-lg">TubeScribe Chatbot</h3>
                <p className="text-xs sm:text-sm text-gray-500 dark:text-gray-400 flex items-center gap-2">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                  </span>
                  Online
                </p>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-grow overflow-y-auto p-4 space-y-4 bg-white dark:bg-gray-900">
              {guideMessages.map((msg) => (
                <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  {msg.role === 'model' && (
                    <div className="w-8 h-8 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center text-gray-500 dark:text-gray-400 mr-2 flex-shrink-0 mt-2">
                      <Bot size={16} />
                    </div>
                  )}
                  <div className={`max-w-[85%] px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm ${msg.role === 'user'
                    ? 'bg-blue-600 text-white rounded-br-sm'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 rounded-bl-sm border border-transparent dark:border-gray-700'
                    }`}>
                    <ReactMarkdown
                      className={`prose prose-sm max-w-none ${msg.role === 'user' ? 'prose-invert' : 'dark:prose-invert'}`}
                      remarkPlugins={[remarkGfm]}
                    >
                      {msg.text}
                    </ReactMarkdown>
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex justify-start">
                  <div className="w-8 h-8 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center text-gray-500 dark:text-gray-400 mr-2 flex-shrink-0 mt-2">
                    <Bot size={16} />
                  </div>
                  <div className="bg-gray-100 dark:bg-gray-800 px-4 py-3 rounded-2xl rounded-bl-sm text-gray-500 dark:text-gray-400 flex items-center gap-2 border border-transparent dark:border-gray-700">
                    <span className="w-1.5 h-1.5 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce"></span>
                    <span className="w-1.5 h-1.5 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce delay-100"></span>
                    <span className="w-1.5 h-1.5 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce delay-200"></span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <form onSubmit={handleSend} className="p-4 bg-white dark:bg-gray-900 border-t border-gray-100 dark:border-gray-800">
              <div className="relative flex items-center bg-gray-50 dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-transparent transition-all">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask me anything..."
                  className="w-full pl-4 pr-12 py-3 bg-transparent border-none outline-none text-sm text-gray-800 dark:text-gray-200 placeholder-gray-400 dark:placeholder-gray-500"
                />
                <button
                  type="submit"
                  disabled={!input.trim() || loading}
                  className="absolute right-2 p-2 bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 rounded-lg shadow-sm border border-gray-100 dark:border-gray-600 hover:bg-blue-50 dark:hover:bg-gray-600 disabled:opacity-50 transition-colors"
                >
                  <Send size={18} />
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Menu;