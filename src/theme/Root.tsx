/**
 * Root component wrapper for Docusaurus
 * This wraps the entire application and adds global components
 */

import React, { useState, useEffect } from 'react';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import ChatbotWidget from '../components/ChatbotWidget';

export default function Root({ children }) {
  const [chapterContext, setChapterContext] = useState<string | undefined>(undefined);

  // Only get chapter context on client side
  useEffect(() => {
    if (ExecutionEnvironment.canUseViewport) {
      const path = window.location.pathname;
      const match = path.match(/\/docs\/(module-\d+\/[^/]+)/);
      setChapterContext(match ? match[1] : undefined);
    }
  }, []);

  return (
    <>
      {children}
      {ExecutionEnvironment.canUseDOM && (
        <ChatbotWidget chapterContext={chapterContext} />
      )}
    </>
  );
}
