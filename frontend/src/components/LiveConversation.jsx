import React, { useState } from 'react';
import { MessageSquare, Sparkles, User, Bot, CheckCircle2, AlertTriangle, ChevronDown, ChevronUp, Send, Globe } from 'lucide-react';

const SAMPLE_QUERIES = [
  { label: 'Corporation?', text: 'What is a corporation?' },
  { label: 'Rachel Carson (Hinglish)', text: 'Rachel Carson ne Obligation to Endure kyu likha tha?' },
  { label: 'वस्तु विनिमय (Barter)', text: 'वस्तु विनिमय में पहला क्या था' },
  { label: 'महाराष्ट्र राजधानी (मराठी)', text: 'महाराष्ट्र राज्याची राजधानी कोणती आहे?' },
  { label: 'தலைநகரம் (தமிழ்)', text: 'இந்தியாவின் தலைநகரம் எது?' },
  { label: 'রাজধানী (বাংলা)', text: 'ভারতের राजधानी কি?' },
];

export default function LiveConversation({
  partialTranscript,
  finalTranscript,
  detectedLanguage,
  systemState,
  streamingAnswer,
  finalAnswer,
  statusResult,
  refusalReason,
  groundednessScore,
  retrievedChunks,
  executeQuery,
}) {
  const [inputText, setInputText] = useState('');
  const [showCitations, setShowCitations] = useState(false);

  const handleTextSubmit = (e) => {
    e.preventDefault();
    if (inputText.trim()) {
      executeQuery(inputText.trim());
      setInputText('');
    }
  };

  const handleChipClick = (queryText) => {
    executeQuery(queryText);
  };

  const hasUserTranscript = Boolean(finalTranscript || partialTranscript);
  const currentAnswer = finalAnswer || streamingAnswer;
  const isGenerating = systemState === 'GENERATING';
  const isListening = systemState === 'LISTENING';
  const isRefusal = statusResult && statusResult.startsWith('refusal');

  return (
    <div className="ui-card col-transcript">
      <div className="card-header">
        <div className="card-title-group">
          <MessageSquare size={14} />
          <span>Live Conversation</span>
        </div>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          {isListening ? 'Streaming Speech' : isGenerating ? 'Streaming LLM' : 'Ready'}
        </span>
      </div>

      {/* Try Asking Suggestions Cloud */}
      <div className="try-asking-section">
        <div className="try-asking-label">
          <Sparkles size={11} />
          <span>Try Asking</span>
        </div>
        <div className="chips-cloud">
          {SAMPLE_QUERIES.map((q, idx) => (
            <button
              key={idx}
              className="query-chip-btn"
              onClick={() => handleChipClick(q.text)}
              title={q.text}
            >
              {q.label}
            </button>
          ))}
        </div>
      </div>

      {/* Conversation Message Stream */}
      <div className="conversation-scroll-area" aria-live="polite" aria-label="Conversation stream">
        {/* User Query Bubble */}
        {hasUserTranscript ? (
          <div className="chat-bubble-user">
            <div className="bubble-meta-user">
              <User size={12} />
              <span>User Query</span>
              {detectedLanguage && detectedLanguage.label && !isListening && (
                <span className="detected-pill-light">
                  <Globe size={9} />
                  <span>Detected · {detectedLanguage.label}</span>
                </span>
              )}
            </div>
            <div className="bubble-text">
              {finalTranscript && <span>{finalTranscript}</span>}
              {partialTranscript && (
                <span className="bubble-text-partial">
                  {finalTranscript ? ` ${partialTranscript}` : partialTranscript}
                </span>
              )}
            </div>
          </div>
        ) : isListening ? (
          <div className="chat-bubble-user">
            <div className="bubble-meta-user">
              <User size={12} />
              <span>Live Listening...</span>
            </div>
            <div className="bubble-text-partial">Listening for speech input...</div>
          </div>
        ) : null}

        {/* Assistant Response Bubble */}
        {currentAnswer ? (
          <div className="chat-bubble-assistant">
            <div className="bubble-meta-assistant">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <Bot size={13} />
                <span style={{ fontWeight: 600 }}>Indic Voice Agent</span>
              </div>
              {statusResult === 'success' && (
                <span className="status-badge-pill verified">
                  <CheckCircle2 size={11} />
                  <span>Verified ({groundednessScore ? (groundednessScore * 100).toFixed(0) : '100'}%)</span>
                </span>
              )}
              {isRefusal && (
                <span className="status-badge-pill refusal">
                  <AlertTriangle size={11} />
                  <span>Refusal Intercepted</span>
                </span>
              )}
            </div>

            <div className="bubble-text">{currentAnswer}</div>

            {/* Expandable Citations / Retrieved Context */}
            {retrievedChunks && retrievedChunks.length > 0 && statusResult === 'success' && (
              <div>
                <button
                  className="citation-toggle-btn"
                  onClick={() => setShowCitations(!showCitations)}
                  aria-expanded={showCitations}
                  aria-controls="live-grounding-citations"
                >
                  <span>{retrievedChunks.length} Grounding Passage{retrievedChunks.length > 1 ? 's' : ''}</span>
                  {showCitations ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                </button>
                {showCitations && (
                  <div className="citation-box" id="live-grounding-citations">
                    {retrievedChunks.map((chunk, idx) => (
                      <div key={idx} style={{ marginBottom: idx < retrievedChunks.length - 1 ? '0.4rem' : 0 }}>
                        <span style={{ fontWeight: 600, fontFamily: 'var(--font-mono)', fontSize: '0.65rem' }}>
                          [{idx + 1}] Score: {chunk.score ? chunk.score.toFixed(3) : 'N/A'}:{' '}
                        </span>
                        <span>{chunk.text}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ) : isGenerating ? (
          <div className="chat-bubble-assistant">
            <div className="bubble-meta-assistant">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <Bot size={13} />
                <span style={{ fontWeight: 600 }}>Indic Voice Agent</span>
              </div>
              <span className="status-badge-pill" style={{ background: 'var(--blue-50)', color: 'var(--blue-500)' }}>
                <Sparkles size={11} />
                <span>Generating...</span>
              </span>
            </div>
            <div className="bubble-text-partial">Generating grounded response from retrieved passages...</div>
          </div>
        ) : null}
      </div>

      {/* Keyboard Text Input Fallback Bar */}
      <form onSubmit={handleTextSubmit} className="chat-input-bar">
        <input
          type="text"
          className="chat-input-field"
          placeholder="Ask in Hindi, English, Hinglish, Marathi, Tamil, Bengali..."
          aria-label="Ask a question across Indic languages or English"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          disabled={systemState === 'LISTENING' || isGenerating}
        />
        <button
          type="submit"
          className="chat-submit-btn"
          disabled={!inputText.trim() || systemState === 'LISTENING' || isGenerating}
          title="Send text query"
          aria-label="Send query"
        >
          <Send size={14} />
        </button>
      </form>
    </div>
  );
}
