import React, { useState, useEffect } from 'react';
import { Youtube, AlertCircle, ExternalLink } from 'lucide-react';

interface YouTubeInputProps {
  url: string;
  onUrlChange: (url: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
}

const YouTubeInput: React.FC<YouTubeInputProps> = ({ url, onUrlChange, onSubmit, isLoading }) => {
  const [videoId, setVideoId] = useState<string | null>(null);

  useEffect(() => {
    const extractVideoId = (inputUrl: string) => {
      const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
      const match = inputUrl.match(regExp);
      return (match && match[2].length === 11) ? match[2] : null;
    };
    setVideoId(extractVideoId(url));
  }, [url]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && videoId && !isLoading) {
      onSubmit();
    }
  };

  // Simplified embed URL construction to maximize compatibility and prevent "Error 153"
  // Removing 'origin' and 'enablejsapi' avoids mismatches in sandboxed environments.
  const getEmbedUrl = (id: string) => {
    return `https://www.youtube.com/embed/${id}?rel=0&modestbranding=1&playsinline=1`;
  };

  return (
    <div className="w-full space-y-6">
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-gray-400">
          <Youtube size={28} />
        </div>
        <input
          type="text"
          value={url}
          onChange={(e) => onUrlChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Paste YouTube Link (e.g., https://youtube.com/...)"
          className="w-full pl-14 pr-4 py-4 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-red-500 focus:border-red-500 outline-none transition-all shadow-sm text-lg text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500"
          disabled={isLoading}
        />
        {url && !videoId && (
            <div className="absolute right-4 top-4 text-red-500" title="Invalid YouTube URL">
                <AlertCircle size={24} />
            </div>
        )}
      </div>

      {/* Video Embed Preview */}
      <div className={`bg-black rounded-xl overflow-hidden shadow-md transition-all duration-500 ${videoId ? 'opacity-100' : 'opacity-0 h-0'}`}>
        {videoId && (
          <div className="aspect-video relative w-full">
            <iframe
              src={getEmbedUrl(videoId)}
              title="YouTube video player"
              className="absolute top-0 left-0 w-full h-full"
              // Add "compute-pressure" to the list below:
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share; compute-pressure"
              allowFullScreen
            ></iframe>
          </div>
        )}
      </div>

      {/* Fallback Link */}
      {videoId && (
        <div className="flex justify-end">
           <a 
             href={`https://www.youtube.com/watch?v=${videoId}`} 
             target="_blank" 
             rel="noopener noreferrer"
             className="text-sm text-gray-500 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 flex items-center gap-2 transition-colors font-medium"
           >
             Video not playing? Watch on YouTube <ExternalLink size={14} />
           </a>
        </div>
      )}
    </div>
  );
};

export default YouTubeInput;