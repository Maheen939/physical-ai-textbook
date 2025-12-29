/**
 * Urdu Translation Button Component

 * Allows users to translate chapter content to Urdu.
 */

import React, { useState, useEffect } from 'react';
import styles from './TranslationButton.module.css';

interface TranslationButtonProps {
  chapterId: string;
  originalContent: string;
  onTranslatedContent?: (content: string) => void;
}

type Language = 'en' | 'ur';

export default function TranslationButton({
  chapterId,
  originalContent,
  onTranslatedContent
}: TranslationButtonProps) {
  const [language, setLanguage] = useState<Language>('en');
  const [urduContent, setUrduContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [cached, setCached] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  // Check if content is already translated
  useEffect(() => {
    const checkCache = async () => {
      try {
        const response = await fetch(
          `${API_URL}/api/translate/status?chapter_id=${chapterId}&target_language=ur`
        );
        if (response.ok) {
          const data = await response.json();
          setCached(data.cached);
        }
      } catch (err) {
        console.error('Failed to check translation status:', err);
      }
    };

    checkCache();
  }, [chapterId]);

  const handleTranslate = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/api/translate/chapter`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chapter_id: chapterId,
          chapter_content: originalContent,
          target_language: 'ur',
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Translation failed');
      }

      const data = await response.json();
      setUrduContent(data.translated_content);
      setLanguage('ur');
      setCached(data.cached);

      if (onTranslatedContent) {
        onTranslatedContent(data.translated_content);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Translation failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSwitchToEnglish = () => {
    setLanguage('en');
    if (onTranslatedContent) {
      onTranslatedContent(originalContent);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.languageToggle}>
        <button
          onClick={handleSwitchToEnglish}
          className={`${styles.langButton} ${language === 'en' ? styles.active : ''}`}
        >
          English
        </button>
        <button
          onClick={handleTranslate}
          disabled={loading}
          className={`${styles.langButton} ${language === 'ur' ? styles.active : ''}`}
          dir="rtl"
        >
          اردو
        </button>
      </div>

      {loading && (
        <span className={styles.loading}>
          <span className={styles.spinner}></span>
          Translating...
        </span>
      )}

      {cached && language === 'ur' && (
        <span className={styles.cachedBadge}>From cache</span>
      )}

      {error && <span className={styles.error}>{error}</span>}
    </div>
  );
}
