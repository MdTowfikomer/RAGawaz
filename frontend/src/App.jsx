import React, { useState } from 'react';
import { useVoiceRAG } from './hooks/useVoiceRAG';
import Header from './components/Header';
import LiveConversation from './components/LiveConversation';
import VoiceHero from './components/VoiceHero';
import PerformanceTelemetry from './components/PerformanceTelemetry';
import BenchmarkMatrix from './components/BenchmarkMatrix';

export default function App() {
  const [activeTab, setActiveTab] = useState('voice'); // 'voice' | 'benchmark'

  const {
    systemState,
    selectedLanguage,
    setSelectedLanguage,
    partialTranscript,
    finalTranscript,
    detectedLanguage,
    streamingAnswer,
    finalAnswer,
    statusResult,
    refusalReason,
    groundednessScore,
    retrievedChunks,
    telemetry,
    errorMessage,
    systemHealth,
    benchmarkData,
    toggleRecording,
    executeQuery,
  } = useVoiceRAG();

  return (
    <div className="app-container">
      {/* Editorial Navigation Header */}
      <Header
        systemHealth={systemHealth}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        selectedLanguage={selectedLanguage}
        setSelectedLanguage={setSelectedLanguage}
      />

      {activeTab === 'voice' ? (
        <main>
          {/* Subtitle Banner inspired by Moss UI */}
          <div className="page-subtitle-banner">
            <p className="page-subtitle-text">
              You're talking to a live Indic voice agent powered by BGE-M3 real-time retrieval and 5-stage guardrails.
              Ask anything across English, Hindi, Hinglish, Marathi, Tamil, or Bengali.
            </p>
          </div>

          {/* 3-Zone Card Layout */}
          <div className="voice-grid-3col">
            {/* Zone 1: Live Conversation & Suggestions */}
            <LiveConversation
              partialTranscript={partialTranscript}
              finalTranscript={finalTranscript}
              detectedLanguage={detectedLanguage}
              systemState={systemState}
              streamingAnswer={streamingAnswer}
              finalAnswer={finalAnswer}
              statusResult={statusResult}
              refusalReason={refusalReason}
              groundednessScore={groundednessScore}
              retrievedChunks={retrievedChunks}
              executeQuery={executeQuery}
            />

            {/* Zone 2: Minimalist Voice Agent Hero */}
            <VoiceHero
              systemState={systemState}
              toggleRecording={toggleRecording}
              telemetry={telemetry}
            />

            {/* Zone 3: Real-Time Telemetry & 5-Stage Guardrails */}
            <PerformanceTelemetry
              telemetry={telemetry}
              statusResult={statusResult}
              retrievedChunks={retrievedChunks}
            />
          </div>

          {/* Error Notice */}
          {errorMessage && (
            <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'var(--coral-50)', border: '1px solid var(--coral-100)', borderRadius: 'var(--radius-md)', color: 'var(--coral-600)', fontSize: '0.85rem' }}>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>SYSTEM NOTICE:</strong> {errorMessage}
            </div>
          )}
        </main>
      ) : (
        /* Benchmark Matrix View */
        <main>
          <BenchmarkMatrix benchmarkData={benchmarkData} />
        </main>
      )}
    </div>
  );
}
