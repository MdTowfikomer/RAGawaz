import { useState, useEffect, useRef, useCallback } from 'react';

export const LANGUAGE_LOCALE_MAP = {
  'auto': 'hi-IN', // Interim recognition baseline; final STT uses backend multilingual auto-detection
  'hi-IN': 'hi-IN', // Hindi (हिन्दी)
  'en-IN': 'en-IN', // English (Indian English locale)
  'en-US': 'en-IN', // English
  'hi-EN': 'en-IN', // Hinglish (Latin alphabet speech capture via en-IN)
  'mr-IN': 'mr-IN', // Marathi (मराठी)
  'ta-IN': 'ta-IN', // Tamil (தமிழ்)
  'bn-IN': 'bn-IN', // Bengali (বাংলা)
};

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

export function useVoiceRAG() {
  // State Machine: READY | LISTENING | TRANSCRIBING | RETRIEVING | GENERATING | COMPLETE | REFUSED | ERROR
  const [systemState, setSystemState] = useState('READY');
  const [selectedLanguage, setSelectedLanguage] = useState('auto'); // auto (Default), hi-IN, en-IN, hi-EN, mr-IN, ta-IN, bn-IN

  // Transcript & Language Detection states
  const [partialTranscript, setPartialTranscript] = useState('');
  const [finalTranscript, setFinalTranscript] = useState('');
  const [detectedLanguage, setDetectedLanguage] = useState(null); // { code: 'english', label: 'English', confidence: 0.94 }

  // Response & telemetry states
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [finalAnswer, setFinalAnswer] = useState('');
  const [statusResult, setStatusResult] = useState(null); // 'success', 'refusal_safety', 'refusal_offtopic', 'refusal_insufficient_evidence', 'refusal_ungrounded'
  const [refusalReason, setRefusalReason] = useState(null);
  const [groundednessScore, setGroundednessScore] = useState(1.0);
  const [retrievedChunks, setRetrievedChunks] = useState([]);
  const [telemetry, setTelemetry] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  // System Health
  const [systemHealth, setSystemHealth] = useState(null);
  const [benchmarkData, setBenchmarkData] = useState(null);

  // Web Speech & MediaRecorder refs
  const recognitionRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingStartTimeRef = useRef(null);
  const firstPartialMsRef = useRef(null);

  // Fetch health and benchmarks on load
  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((res) => res.json())
      .then((data) => setSystemHealth(data))
      .catch((err) => console.error('Health check error:', err));

    fetch(`${API_BASE}/api/benchmark/results`)
      .then((res) => res.json())
      .then((data) => setBenchmarkData(data))
      .catch((err) => console.error('Benchmark data error:', err));
  }, []);

  // Initialize and bind Speech Recognition to selectedLanguage (for live interim hints)
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const rec = new SpeechRecognition();
      rec.continuous = true;
      rec.interimResults = true;
      
      const targetLocale = LANGUAGE_LOCALE_MAP[selectedLanguage] || 'hi-IN';
      rec.lang = targetLocale;

      rec.onresult = (event) => {
        if (!firstPartialMsRef.current && recordingStartTimeRef.current) {
          firstPartialMsRef.current = Math.round(performance.now() - recordingStartTimeRef.current);
        }
        let interim = '';
        let final = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const trans = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            final += trans;
          } else {
            interim += trans;
          }
        }
        if (interim) setPartialTranscript(interim);
        if (final) {
          setFinalTranscript((prev) => (prev ? `${prev} ${final}` : final));
          setPartialTranscript('');
        }
      };

      rec.onerror = (event) => {
        console.warn('Speech recognition error:', event.error);
      };

      rec.onend = () => {
        // Handled in stopRecording
      };

      recognitionRef.current = rec;
    }
  }, [selectedLanguage]);

  // Execute Query with SSE Streaming LLM
  const executeQuery = useCallback(async (queryText, langMeta = null, sttLatencyMs = null) => {
    const q = (queryText || finalTranscript || partialTranscript).trim();
    if (!q) return;

    setSystemState('RETRIEVING');
    setFinalTranscript(q);
    setPartialTranscript('');
    setStreamingAnswer('');
    setFinalAnswer('');
    setStatusResult(null);
    setRefusalReason(null);
    setRetrievedChunks([]);
    setErrorMessage(null);
    if (sttLatencyMs !== null) {
      setTelemetry((prev) => ({ ...(prev || {}), stt_first_partial_ms: sttLatencyMs }));
    }
    if (langMeta) {
      setDetectedLanguage(langMeta);
    }

    try {
      const response = await fetch(`${API_BASE}/api/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, top_k: 5 }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let accumulatedAnswer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;
          let eventType = 'message';
          let dataJson = null;

          const eventMatch = line.match(/^event:\s*(.+)$/m);
          if (eventMatch) eventType = eventMatch[1].trim();

          const dataMatch = line.match(/^data:\s*(.+)$/m);
          if (dataMatch) {
            try {
              dataJson = JSON.parse(dataMatch[1].trim());
            } catch (e) {
              dataJson = dataMatch[1].trim();
            }
          }

          if (eventType === 'status') {
            if (dataJson?.state === 'GENERATING') {
              setSystemState('GENERATING');
              if (dataJson.retrieved_chunks) setRetrievedChunks(dataJson.retrieved_chunks);
              if (dataJson.metrics) {
                setTelemetry((prev) => ({
                  ...(dataJson.metrics || {}),
                  stt_first_partial_ms: prev?.stt_first_partial_ms ?? dataJson.metrics?.stt_first_partial_ms ?? null,
                }));
              }
            }
          } else if (eventType === 'token') {
            const token = dataJson?.delta || '';
            accumulatedAnswer += token;
            setStreamingAnswer(accumulatedAnswer);
          } else if (eventType === 'refusal') {
            setSystemState('REFUSED');
            setStatusResult(dataJson.status);
            setRefusalReason(dataJson.refusal_reason);
            setFinalAnswer(dataJson.answer || accumulatedAnswer);
            if (dataJson.retrieved_chunks) setRetrievedChunks(dataJson.retrieved_chunks);
            if (dataJson.metrics) {
              setTelemetry((prev) => ({
                ...(dataJson.metrics || {}),
                stt_first_partial_ms: prev?.stt_first_partial_ms ?? dataJson.metrics?.stt_first_partial_ms ?? null,
              }));
            }
            if (dataJson.groundedness_score !== undefined) setGroundednessScore(dataJson.groundedness_score);
            return;
          } else if (eventType === 'complete') {
            setSystemState('COMPLETE');
            setStatusResult('success');
            setFinalAnswer(dataJson.answer || accumulatedAnswer);
            if (dataJson.retrieved_chunks) setRetrievedChunks(dataJson.retrieved_chunks);
            if (dataJson.metrics) {
              setTelemetry((prev) => ({
                ...(dataJson.metrics || {}),
                stt_first_partial_ms: prev?.stt_first_partial_ms ?? dataJson.metrics?.stt_first_partial_ms ?? null,
              }));
            }
            if (dataJson.groundedness_score !== undefined) setGroundednessScore(dataJson.groundedness_score);
            return;
          } else if (eventType === 'error') {
            setSystemState('ERROR');
            setErrorMessage(dataJson?.error || 'An error occurred during query generation.');
            return;
          }
        }
      }

      // If finished stream without explicit terminal event
      setSystemState('COMPLETE');
      setStatusResult('success');
      setFinalAnswer(accumulatedAnswer);
    } catch (err) {
      console.warn('SSE Streaming connection failed, falling back to REST /api/query:', err);
      // Resilient Fallback to REST endpoint
      try {
        const res = await fetch(`${API_BASE}/api/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: q, top_k: 5 }),
        });
        const data = await res.json();
        setFinalAnswer(data.answer);
        setStatusResult(data.status);
        setRefusalReason(data.refusal_reason);
        setRetrievedChunks(data.retrieved_chunks || []);
        setTelemetry(data.telemetry || {});
        if (data.groundedness_score !== undefined) setGroundednessScore(data.groundedness_score);

        if (data.status && data.status.startsWith('refusal')) {
          setSystemState('REFUSED');
        } else {
          setSystemState('COMPLETE');
        }
      } catch (fallbackErr) {
        console.error('Fallback query failed:', fallbackErr);
        setSystemState('ERROR');
        setErrorMessage('Failed to connect to Voice RAG backend.');
      }
    }
  }, [finalTranscript, partialTranscript]);

  // Start Recording
  const startRecording = useCallback(() => {
    setSystemState('LISTENING');
    setPartialTranscript('');
    setFinalTranscript('');
    setDetectedLanguage(null);
    setStreamingAnswer('');
    setFinalAnswer('');
    setErrorMessage(null);
    recordingStartTimeRef.current = performance.now();
    firstPartialMsRef.current = null;

    // Start browser recognition for every mode so the transcript remains available
    // as a fallback when the remote multilingual STT provider is unavailable.
    if (recognitionRef.current) {
      try {
        const targetLocale = LANGUAGE_LOCALE_MAP[selectedLanguage] || 'hi-IN';
        recognitionRef.current.lang = targetLocale;
        recognitionRef.current.start();
      } catch (e) {
        console.warn('SpeechRecognition start warning:', e);
      }
    }

    // MediaRecorder for capturing real audio waveform for Multilingual STT & Auto-Detection
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then((stream) => {
          const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
            ? 'audio/webm;codecs=opus'
            : '';
          const mediaRecorder = mimeType
            ? new MediaRecorder(stream, { mimeType })
            : new MediaRecorder(stream);
          mediaRecorderRef.current = mediaRecorder;
          audioChunksRef.current = [];

          mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
              audioChunksRef.current.push(event.data);
            }
          };

          mediaRecorder.start();
        })
        .catch((err) => {
          console.warn('Microphone stream error:', err);
        });
    }
  }, [selectedLanguage]);

  // Stop Recording
  const stopRecording = useCallback(() => {
    setSystemState('TRANSCRIBING');

    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (e) {
        console.warn('SpeechRecognition stop warning:', e);
      }
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.onstop = async () => {
        const recordingType = mediaRecorderRef.current?.mimeType || 'audio/webm';
        const audioBlob = new Blob(audioChunksRef.current, { type: recordingType });
        const fileExtension = recordingType.includes('webm')
          ? 'webm'
          : recordingType.includes('mp4')
            ? 'mp4'
            : 'wav';
        // Stop all tracks
        mediaRecorderRef.current?.stream?.getTracks().forEach((track) => track.stop());

        if (audioBlob.size > 0) {
          // Multilingual STT & Language Auto-Detection via backend
          try {
            const formData = new FormData();
            formData.append('file', audioBlob, `recording.${fileExtension}`);
            formData.append('language_code', selectedLanguage);
            const res = await fetch(`${API_BASE}/api/voice/stt`, {
              method: 'POST',
              body: formData,
            });
            const data = await res.json();
            const transcript = data.text || '';
            const sttMs = data.stt_latency_ms || firstPartialMsRef.current || null;
            const langMeta = {
              code: data.detected_language || 'unknown',
              label: data.language_display || 'Unknown',
              confidence: data.language_confidence,
            };

            if (transcript) {
              setFinalTranscript(transcript);
              setDetectedLanguage(langMeta);
              executeQuery(transcript, langMeta, sttMs);
            } else {
              // Fallback to browser transcript if backend STT was empty
              const capturedText = (finalTranscript || partialTranscript).trim();
              if (capturedText) {
                executeQuery(capturedText, null, firstPartialMsRef.current);
              } else {
                setSystemState('READY');
              }
            }
          } catch (err) {
            console.error('Voice STT process error:', err);
            const capturedText = (finalTranscript || partialTranscript).trim();
            if (capturedText) {
              executeQuery(capturedText, null, firstPartialMsRef.current);
            } else {
              setSystemState('READY');
            }
          }
        } else {
          const capturedText = (finalTranscript || partialTranscript).trim();
          if (capturedText) {
            executeQuery(capturedText, null, firstPartialMsRef.current);
          } else {
            setSystemState('READY');
          }
        }
      };

      mediaRecorderRef.current.stop();
    } else {
      // Fallback when MediaRecorder is not active
      setTimeout(() => {
        const capturedText = (finalTranscript || partialTranscript).trim();
        if (capturedText) {
          executeQuery(capturedText, null, firstPartialMsRef.current);
        } else {
          setSystemState('READY');
        }
      }, 300);
    }
  }, [finalTranscript, partialTranscript, selectedLanguage, executeQuery]);

  const toggleRecording = useCallback(() => {
    if (systemState === 'LISTENING') {
      stopRecording();
    } else {
      startRecording();
    }
  }, [systemState, startRecording, stopRecording]);

  return {
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
    stopRecording,
    executeQuery,
  };
}
