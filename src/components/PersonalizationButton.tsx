/**
 * Content Personalization Button Component

 * Allows users to personalize chapter content based on their profile.
 */

import React, { useState, useEffect } from 'react';
import { useAuth } from './auth/AuthProvider';
import styles from './PersonalizationButton.module.css';

interface PersonalizationButtonProps {
  chapterId: string;
  originalContent: string;
  onPersonalizedContent?: (content: string) => void;
}

export default function PersonalizationButton({
  chapterId,
  originalContent,
  onPersonalizedContent
}: PersonalizationButtonProps) {
  const { user, isAuthenticated } = useAuth();
  const [personalizedContent, setPersonalizedContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [cached, setCached] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  // Check if content is already personalized
  useEffect(() => {
    const checkCache = async () => {
      if (!isAuthenticated || !user) return;

      try {
        const response = await fetch(
          `${API_URL}/api/personalize/status?user_id=${user.id}&chapter_id=${chapterId}`
        );
        if (response.ok) {
          const data = await response.json();
          setCached(data.cached);
        }
      } catch (err) {
        console.error('Failed to check personalization status:', err);
      }
    };

    checkCache();
  }, [chapterId, isAuthenticated, user]);

  const handlePersonalize = async () => {
    if (!isAuthenticated) {
      alert('Please sign in to personalize content');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/api/personalize/chapter`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chapter_id: chapterId,
          user_id: user?.id,
          chapter_content: originalContent,
          profile: user?.profile || {},
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Personalization failed');
      }

      const data = await response.json();
      setPersonalizedContent(data.personalized_content);
      setCached(data.cached);

      if (onPersonalizedContent) {
        onPersonalizedContent(data.personalized_content);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to personalize content');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setPersonalizedContent(null);
    setCached(false);
    if (onPersonalizedContent) {
      onPersonalizedContent(originalContent);
    }
  };

  if (!isAuthenticated) {
    return (
      <div className={styles.container} title="Sign in to enable personalization">
        <button disabled className={styles.button}>
          Personalize
        </button>
        <span className={styles.hint}>Sign in to personalize</span>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {personalizedContent ? (
        <>
          <button onClick={handleReset} className={styles.resetButton}>
            Show Original
          </button>
          {cached && <span className={styles.cachedBadge}>From cache</span>}
        </>
      ) : (
        <button
          onClick={handlePersonalize}
          disabled={loading}
          className={styles.button}
        >
          {loading ? 'Personalizing...' : 'Personalize Content'}
        </button>
      )}

      {error && <span className={styles.error}>{error}</span>}
    </div>
  );
}
