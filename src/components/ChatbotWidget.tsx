/**
 * Chatbot Widget Component
 * Floating chat interface with RAG-powered responses
 */

import React, { useState, useRef, useEffect } from 'react';
import { useChatbot, Message } from '../hooks/useChatbot';
import styles from './ChatbotWidget.module.css';

interface ChatbotWidgetProps {
  chapterContext?: string;
}

export const ChatbotWidget: React.FC<ChatbotWidgetProps> = ({ chapterContext }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { messages, isLoading, error, sendMessage, clearChat } = useChatbot();

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const message = inputValue.trim();
    setInputValue('');
    await sendMessage(message, chapterContext);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const getSelectedText = () => {
    const selection = window.getSelection();
    return selection?.toString() || '';
  };

  const handleAskAboutSelection = () => {
    const selectedText = getSelectedText();
    if (selectedText) {
      setInputValue(`Explain this: "${selectedText}"`);
      setIsOpen(true);
    }
  };

  return (
    <>
      {/* Floating Chat Button */}
      <button
        className={styles.chatButton}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle chatbot"
        title="AI Tutor - Ask me anything!"
      >
        {isOpen ? '✕' : '💬'}
      </button>

      {/* Ask About Selection Button (shows when text is selected) */}
      {window.getSelection()?.toString() && !isOpen && (
        <button
          className={styles.selectionButton}
          onClick={handleAskAboutSelection}
          title="Ask chatbot about selected text"
        >
          🤔 Ask AI
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div className={styles.chatWindow}>
          {/* Header */}
          <div className={styles.chatHeader}>
            <div className={styles.headerContent}>
              <span className={styles.headerIcon}>🤖</span>
              <div>
                <h3 className={styles.headerTitle}>AI Tutor</h3>
                <p className={styles.headerSubtitle}>
                  {chapterContext ? `Context: ${chapterContext}` : 'Ask me anything about the course!'}
                </p>
              </div>
            </div>
            <button
              className={styles.clearButton}
              onClick={clearChat}
              title="Clear chat"
            >
              🗑️
            </button>
          </div>

          {/* Messages */}
          <div className={styles.messagesContainer}>
            {messages.length === 0 && (
              <div className={styles.welcomeMessage}>
                <h4>👋 Welcome!</h4>
                <p>I'm your AI tutor for Physical AI & Humanoid Robotics.</p>
                <p>Ask me anything about:</p>
                <ul>
                  <li>ROS 2 concepts</li>
                  <li>Gazebo simulation</li>
                  <li>NVIDIA Isaac</li>
                  <li>Humanoid robotics</li>
                  <li>Code examples</li>
                </ul>
                <p>
                  <strong>Tip:</strong> Select any text on the page and click "Ask AI" to ask
                  questions about it!
                </p>
              </div>
            )}

            {messages.map((msg, index) => (
              <ChatMessage key={index} message={msg} />
            ))}

            {isLoading && (
              <div className={styles.messageUser}>
                <div className={styles.messageContent}>
                  <div className={styles.typingIndicator}>
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className={styles.errorMessage}>
                ⚠️ {error}
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className={styles.inputContainer}>
            <textarea
              className={styles.input}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question..."
              rows={2}
              disabled={isLoading}
            />
            <button
              className={styles.sendButton}
              onClick={handleSend}
              disabled={!inputValue.trim() || isLoading}
            >
              {isLoading ? '⏳' : '➤'}
            </button>
          </div>
        </div>
      )}
    </>
  );
};

/**
 * Individual chat message component
 */
const ChatMessage: React.FC<{ message: Message }> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div className={isUser ? styles.messageUser : styles.messageAssistant}>
      <div className={styles.messageContent}>
        <div className={styles.messageText}>
          {message.content}
        </div>

        {message.sources && message.sources.length > 0 && (
          <div className={styles.sources}>
            <details>
              <summary>📚 Sources ({message.sources.length})</summary>
              <ul>
                {message.sources.map((source, idx) => (
                  <li key={idx}>{source}</li>
                ))}
              </ul>
            </details>
          </div>
        )}

        <div className={styles.messageTime}>
          {message.timestamp.toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </div>
      </div>
    </div>
  );
};

export default ChatbotWidget;
