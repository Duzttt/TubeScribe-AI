export enum ProcessingStatus {
  IDLE = 'IDLE',
  PROCESSING = 'PROCESSING',
  COMPLETED = 'COMPLETED',
  ERROR = 'ERROR',
}

export enum ContentType {
  TRANSCRIPT = 'TRANSCRIPT',
  SUMMARY = 'SUMMARY',
  TRANSLATION = 'TRANSLATION',
  TRANSLATED_TRANSCRIPT = 'TRANSLATED_TRANSCRIPT',
  CHAT = 'CHAT',
}

export interface AnalysisResult {
  transcript?: string;
  summary?: string;
  translation?: string;
  targetLanguage?: string;
  keywords?: string[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  text: string;
  timestamp: number;
}

export enum TargetLanguage {
  ORIGINAL = 'Original (Auto-Detect)',
  ENGLISH = 'English',
  SPANISH = 'Spanish',
  FRENCH = 'French',
  GERMAN = 'German',
  CHINESE_SIMPLIFIED = 'Chinese (Simplified)',
  JAPANESE = 'Japanese',
  KOREAN = 'Korean',
  HINDI = 'Hindi',
  ARABIC = 'Arabic',
  PORTUGUESE = 'Portuguese',
  ITALIAN = 'Italian',
  RUSSIAN = 'Russian',
  DUTCH = 'Dutch',
  TURKISH = 'Turkish',
  VIETNAMESE = 'Vietnamese'
}