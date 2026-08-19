import React, { useState } from 'react';
import { Activity, Zap, Cpu, Sparkles, Mic, Shield, Filter, FileSearch, CheckCircle2, AlertTriangle, ChevronDown, ChevronUp, Layers, Timer, CheckSquare, XCircle } from 'lucide-react';

export default function PerformanceTelemetry({
  telemetry,
  statusResult,
  retrievedChunks,
}) {
  const [showPassages, setShowPassages] = useState(false);

  const embeddingMs = telemetry?.embedding_ms ?? telemetry?.query_embedding_ms ?? null;
  const faissMs = telemetry?.faiss_ms ?? null;
  const bm25Ms = telemetry?.bm25_ms ?? null;
  const rrfMs = telemetry?.rrf_ms ?? null;
  const preLlmTotalMs = telemetry?.pre_llm_total_ms ?? null;
  const llmTtftMs = telemetry?.llm_ttft_ms ?? null;
  const textToAnswerMs = telemetry?.text_to_answer_ms ?? telemetry?.harness_ms ?? null;
  const sttPartialMs = telemetry?.stt_first_partial_ms ?? null;

  // Boundary Decision Diagnostics
  const entityMatch = telemetry?.entity_match || (statusResult === 'refusal_insufficient_evidence' ? 'FAIL' : 'PASS');
  const evidenceStatus = telemetry?.evidence_status || (['refusal_offtopic', 'refusal_insufficient_evidence'].includes(statusResult) ? 'INSUFFICIENT' : 'SUFFICIENT');
  const llmInvocation = telemetry?.llm_invocation || (['refusal_safety', 'refusal_offtopic', 'refusal_insufficient_evidence'].includes(statusResult) ? 'SKIPPED' : 'EXECUTED');
  const groundednessVerdict = telemetry?.groundedness_verdict || (statusResult === 'success' ? 'VERIFIED' : (llmInvocation === 'SKIPPED' ? 'SKIPPED' : 'UNVERIFIED'));

  return (
    <div className="ui-card col-telemetry">
      <div className="card-header">
        <div className="card-title-group">
          <Activity size={14} />
          <span>Real-Time Boundary Telemetry</span>
        </div>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          BGE-M3 + HNSW + BM25 + Groq
        </span>
      </div>

      {/* Discrete Latency Breakdown Table */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gap: '0.4rem',
        marginBottom: '0.75rem',
      }}>
        {sttPartialMs !== null && (
          <div className="metric-mini-box">
            <div className="metric-mini-title">STT First Partial</div>
            <div className="metric-mini-value" style={{ color: 'var(--blue-500)' }}>
              {sttPartialMs.toFixed(1)} ms
            </div>
          </div>
        )}

        <div className="metric-mini-box">
          <div className="metric-mini-title">BGE-M3 Embed</div>
          <div className="metric-mini-value" style={{ color: 'var(--text-primary)' }}>
            {embeddingMs !== null ? `${embeddingMs.toFixed(1)} ms` : '—'}
          </div>
        </div>

        <div className="metric-mini-box">
          <div className="metric-mini-title">FAISS HNSW (Top 50)</div>
          <div className="metric-mini-value" style={{ color: 'var(--emerald-600)' }}>
            {faissMs !== null ? `${faissMs.toFixed(2)} ms` : '—'}
          </div>
        </div>

        <div className="metric-mini-box">
          <div className="metric-mini-title">BM25 Sparse (Top 50)</div>
          <div className="metric-mini-value" style={{ color: 'var(--indigo-600)' }}>
            {bm25Ms !== null ? `${bm25Ms.toFixed(2)} ms` : '—'}
          </div>
        </div>

        <div className="metric-mini-box">
          <div className="metric-mini-title">RRF Fusion (k=60)</div>
          <div className="metric-mini-value" style={{ color: 'var(--amber-600)' }}>
            {rrfMs !== null ? `${rrfMs.toFixed(2)} ms` : '—'}
          </div>
        </div>

        <div className="metric-mini-box" style={{ background: 'var(--bg-subtle)', border: '1px solid var(--border-color)' }}>
          <div className="metric-mini-title" style={{ fontWeight: 700 }}>Pre-LLM Boundary</div>
          <div className="metric-mini-value" style={{ color: 'var(--coral-600)', fontWeight: 700 }}>
            {preLlmTotalMs !== null ? `${preLlmTotalMs.toFixed(1)} ms` : '—'}
          </div>
        </div>

        {llmInvocation === 'EXECUTED' && (
          <div className="metric-mini-box">
            <div className="metric-mini-title">LLM TTFT (Groq)</div>
            <div className="metric-mini-value" style={{ color: 'var(--blue-600)' }}>
              {llmTtftMs !== null ? `${llmTtftMs.toFixed(1)} ms` : '—'}
            </div>
          </div>
        )}

        <div className="metric-mini-box" style={{ background: 'var(--card-hover)', border: '1px solid var(--border-color)' }}>
          <div className="metric-mini-title" style={{ fontWeight: 700 }}>Text → Answer Total</div>
          <div className="metric-mini-value" style={{ color: 'var(--text-primary)', fontWeight: 800 }}>
            {textToAnswerMs !== null ? `${textToAnswerMs.toFixed(1)} ms` : '—'}
          </div>
        </div>
      </div>

      {/* Boundary Decision & Verification Panel */}
      <div className="guardrail-pipeline-card" style={{ marginBottom: '0.75rem' }}>
        <div className="guardrail-pipeline-title">
          <Shield size={12} />
          <span>Boundary Decisions & Verification</span>
        </div>

        <div className="pipeline-stage-item">
          <div className="stage-label">
            <FileSearch size={12} />
            <span>Evidence Sufficiency</span>
          </div>
          <span className="stage-status-tag" style={{
            background: evidenceStatus === 'SUFFICIENT' ? 'var(--emerald-100)' : 'var(--coral-100)',
            color: evidenceStatus === 'SUFFICIENT' ? 'var(--emerald-700)' : 'var(--coral-700)',
            fontWeight: 700,
          }}>
            {evidenceStatus}
          </span>
        </div>

        <div className="pipeline-stage-item">
          <div className="stage-label">
            <Filter size={12} />
            <span>Entity Match</span>
          </div>
          <span className="stage-status-tag" style={{
            background: entityMatch === 'PASS' ? 'var(--emerald-100)' : (entityMatch === 'FAIL' ? 'var(--coral-100)' : 'var(--bg-subtle)'),
            color: entityMatch === 'PASS' ? 'var(--emerald-700)' : (entityMatch === 'FAIL' ? 'var(--coral-700)' : 'var(--text-muted)'),
            fontWeight: 700,
          }}>
            {entityMatch}
          </span>
        </div>

        <div className="pipeline-stage-item">
          <div className="stage-label">
            <Cpu size={12} />
            <span>LLM Invocation</span>
          </div>
          <span className="stage-status-tag" style={{
            background: llmInvocation === 'EXECUTED' ? 'var(--blue-100)' : 'var(--amber-100)',
            color: llmInvocation === 'EXECUTED' ? 'var(--blue-700)' : 'var(--amber-700)',
            fontWeight: 700,
          }}>
            {llmInvocation}
          </span>
        </div>

        <div className="pipeline-stage-item">
          <div className="stage-label">
            <CheckCircle2 size={12} />
            <span>Groundedness Verdict</span>
          </div>
          <span className="stage-status-tag" style={{
            background: groundednessVerdict === 'VERIFIED' ? 'var(--emerald-100)' : (groundednessVerdict === 'UNVERIFIED' ? 'var(--coral-100)' : 'var(--bg-subtle)'),
            color: groundednessVerdict === 'VERIFIED' ? 'var(--emerald-700)' : (groundednessVerdict === 'UNVERIFIED' ? 'var(--coral-700)' : 'var(--text-muted)'),
            fontWeight: 700,
          }}>
            {groundednessVerdict}
          </span>
        </div>
      </div>

      {/* Retrieved Context Passages Accordion */}
      {retrievedChunks && retrievedChunks.length > 0 && (
        <div>
          <button
            className="chunks-accordion-header"
            onClick={() => setShowPassages(!showPassages)}
            aria-expanded={showPassages}
            aria-controls="telemetry-retrieved-chunks-list"
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <Layers size={13} />
              <span>Retrieved Chunks ({retrievedChunks.length})</span>
            </div>
            {showPassages ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {showPassages && (
            <div className="chunks-scroll-list" id="telemetry-retrieved-chunks-list">
              {retrievedChunks.map((chunk, idx) => (
                <div key={idx} className="chunk-mini-card">
                  <div className="chunk-mini-header">
                    <span>[{idx + 1}] {chunk.chunk_id || `chunk_${idx}`}</span>
                    <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>
                      Score: {chunk.score ? chunk.score.toFixed(3) : 'N/A'}
                    </span>
                  </div>
                  <div className="chunk-mini-text">{chunk.text}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

