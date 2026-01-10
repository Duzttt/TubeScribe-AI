import React, { useState, useEffect } from 'react';
import { Copy, Check, Download, Eye, FileCode } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';

interface ResultCardProps {
  title: string;
  content: string;
  type: 'text' | 'markdown';
  icon: React.ReactNode;
  isLoading?: boolean;
}

const ResultCard: React.FC<ResultCardProps> = ({ title, content, type, icon, isLoading }) => {
  const [copied, setCopied] = useState(false);
  const [viewMode, setViewMode] = useState<'preview' | 'raw'>(type === 'markdown' ? 'preview' : 'raw');

  useEffect(() => {
    setViewMode(type === 'markdown' ? 'preview' : 'raw');
  }, [type]);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title.toLowerCase().replace(/\s+/g, '-')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const toggleViewMode = () => {
    setViewMode(prev => prev === 'preview' ? 'raw' : 'preview');
  };

  if (!content && !isLoading) return null;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden flex flex-col h-full animate-fade-in-up transition-colors">
      <div className="px-6 py-5 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center bg-gray-50/50 dark:bg-gray-900/30">
        <div className="flex items-center gap-3 text-gray-800 dark:text-gray-200 font-bold text-xl">
          {icon}
          <h3>{title}</h3>
        </div>
        <div className="flex items-center gap-3">
          <button
             onClick={toggleViewMode}
             disabled={!content || isLoading}
             className="p-2.5 text-gray-500 dark:text-gray-400 hover:text-purple-600 dark:hover:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/30 rounded-lg transition-colors"
             title={viewMode === 'preview' ? "Show Raw Text" : "Show Markdown Preview"}
           >
             {viewMode === 'preview' ? <FileCode size={22} /> : <Eye size={22} />}
           </button>
           <button
            onClick={handleDownload}
            disabled={!content || isLoading}
            className="p-2.5 text-gray-500 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/30 rounded-lg transition-colors"
            title="Download as .txt"
          >
            <Download size={22} />
          </button>
          <button
            onClick={handleCopy}
            disabled={!content || isLoading}
            className="p-2.5 text-gray-500 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 hover:bg-green-50 dark:hover:bg-green-900/30 rounded-lg transition-colors"
            title="Copy to clipboard"
          >
            {copied ? <Check size={22} /> : <Copy size={22} />}
          </button>
        </div>
      </div>
      
      <div className="p-8 flex-grow overflow-y-auto max-h-[800px] relative bg-white dark:bg-gray-800 transition-colors">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center h-48 space-y-6">
             <div className="w-10 h-10 border-4 border-blue-200 dark:border-blue-800 border-t-blue-600 dark:border-t-blue-400 rounded-full animate-spin"></div>
             <p className="text-lg text-gray-500 dark:text-gray-400 animate-pulse">Generating {title.toLowerCase()}...</p>
          </div>
        ) : (
          viewMode === 'preview' ? (
            /* HUGE TEXT BUMP: prose-xl */
            <ReactMarkdown 
              className="prose prose-xl prose-slate dark:prose-invert max-w-none text-gray-700 dark:text-gray-300 leading-relaxed" 
              remarkPlugins={[remarkGfm, remarkBreaks]}
            >
              {content}
            </ReactMarkdown>
          ) : (
            <div className="prose prose-xl prose-slate dark:prose-invert max-w-none text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-wrap font-mono text-base sm:text-lg">
               {content}
            </div>
          )
        )}
      </div>
    </div>
  );
};

export default ResultCard;