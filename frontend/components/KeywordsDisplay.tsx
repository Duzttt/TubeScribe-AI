import React from 'react';
import { Tag } from 'lucide-react';

interface KeywordsDisplayProps {
  keywords: string[];
  className?: string;
}

const KeywordsDisplay: React.FC<KeywordsDisplayProps> = ({ keywords, className = '' }) => {
  if (!keywords || keywords.length === 0) {
    return null;
  }

  return (
    <div className={`bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-xl p-4 sm:p-5 border border-purple-200 dark:border-purple-800 ${className}`}>
      <div className="flex items-center gap-2 mb-3">
        <Tag size={18} className="text-purple-600 dark:text-purple-400" />
        <h4 className="font-bold text-sm sm:text-base text-purple-700 dark:text-purple-300">
          Key Topics
        </h4>
        <span className="text-xs text-purple-600 dark:text-purple-400 font-medium">
          ({keywords.length})
        </span>
      </div>
      <div className="flex flex-wrap gap-2">
        {keywords.map((keyword, index) => (
          <span
            key={index}
            className="px-3 py-1.5 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg text-sm font-medium border border-purple-200 dark:border-purple-700 shadow-sm hover:shadow-md transition-shadow"
          >
            {keyword}
          </span>
        ))}
      </div>
    </div>
  );
};

export default KeywordsDisplay;