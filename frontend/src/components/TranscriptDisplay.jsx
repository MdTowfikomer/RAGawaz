import React from 'react';

export default function TranscriptDisplay({
  partialTranscript,
  finalTranscript,
  detectedLanguage,
  systemState,
}) {
  const hasContent = Boolean(finalTranscript || partialTranscript);

  if (!hasContent && systemState === 'READY') {
    return null;
  }

  return (
    <div className="transcript-section">
      <div className="transcript-header">
        <span>Transcribed Spoken Query</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          {detectedLanguage && detectedLanguage.label && systemState !== 'LISTENING' && (
            <span className="detected-lang-pill">
              Detected · {detectedLanguage.label}
            </span>
          )}
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            {systemState === 'LISTENING' ? 'Live Streaming' : 'Finalized'}
          </span>
        </div>
      </div>
      <div className="transcript-body">
        {finalTranscript && <span>{finalTranscript}</span>}
        {partialTranscript && (
          <span className="transcript-partial">
            {finalTranscript ? ` ${partialTranscript}` : partialTranscript}
          </span>
        )}
        {!hasContent && systemState === 'LISTENING' && (
          <span className="transcript-partial">Listening for speech input...</span>
        )}
      </div>
    </div>
  );
}
