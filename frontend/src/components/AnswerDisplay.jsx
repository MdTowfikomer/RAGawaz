import React from 'react';

export default function AnswerDisplay({
  systemState,
  streamingAnswer,
  finalAnswer,
  statusResult,
  refusalReason,
  groundednessScore,
}) {
  const isGenerating = systemState === 'GENERATING';
  const isRefused = systemState === 'REFUSED' || (statusResult && statusResult.startsWith('refusal'));
  const isComplete = systemState === 'COMPLETE';
  const answerText = finalAnswer || streamingAnswer;

  if (!answerText && !isGenerating && !isRefused) {
    return null;
  }

  // Refusal reason human labels
  const refusalLabels = {
    safety_blocklist_triggered: 'Safety Guardrail Triggered (Sub-1ms Pattern Blocklist)',
    relevance_threshold_not_met: 'Relevance Gate Refusal (Out-of-Domain / Low Similarity)',
    insufficient_confidence_evidence: 'Insufficient Evidence Interceptor (Low Confidence Pre-LLM)',
    inability_stated: 'Grounding Verification (Model Stated Inability)',
    topic_drift_rejected: 'Grounding Verification (Topic Drift Detected)',
    hallucination_detected: 'Groundedness Verifier Refusal (Token Overlap / Hallucination)',
  };

  return (
    <div className="answer-section">
      <div className="answer-header">
        <div className="answer-title">
          <span>{isRefused ? '🛡️ Refusal Notice' : '⚡ Grounded Response'}</span>
        </div>
        <div>
          {isComplete && !isRefused && (
            <span className="grounded-badge badge-grounded">
              VERIFIED • Groundedness {groundednessScore?.toFixed(2) ?? '1.00'}
            </span>
          )}
          {isRefused && (
            <span className="grounded-badge badge-refusal">
              GUARDRAIL BLOCK
            </span>
          )}
          {isGenerating && (
            <span className="grounded-badge" style={{ background: 'rgba(37, 99, 235, 0.1)', color: '#2563EB' }}>
              STREAMING ANSWER...
            </span>
          )}
        </div>
      </div>

      {isRefused ? (
        <div className="refusal-box">
          {refusalReason && (
            <div className="refusal-reason-tag">
              Reason: {refusalLabels[refusalReason] || refusalReason}
            </div>
          )}
          <p>{answerText || 'This query cannot be answered based on the provided corpus.'}</p>
        </div>
      ) : (
        <div className="answer-content">
          {answerText}
          {isGenerating && <span className="blinking-cursor" />}
        </div>
      )}
    </div>
  );
}
