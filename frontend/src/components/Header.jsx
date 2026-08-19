import React from 'react';
import { Mic, Sparkles, Layers, Activity } from 'lucide-react';

export const LANGUAGES = [
  { code: 'auto', label: 'Auto' },
  { code: 'hi-IN', label: 'Hindi (हिन्दी)' },
  { code: 'en-IN', label: 'English' },
  { code: 'hi-EN', label: 'Hinglish' },
  { code: 'mr-IN', label: 'Marathi (मराठी)' },
  { code: 'ta-IN', label: 'Tamil (தமிழ்)' },
  { code: 'bn-IN', label: 'Bengali (বাংলা)' },
];

export default function Header({
  systemHealth,
  activeTab,
  setActiveTab,
  selectedLanguage,
  setSelectedLanguage,
}) {
  const loadedCount = systemHealth?.corpus_stats?.loaded_passages || 301108;
  const embedderModel = systemHealth?.corpus_stats?.embedder || 'BAAI/bge-m3 (1024-d)';

  return (
    <header className="app-header">
      <div className="brand-section">
        <div className="brand-icon-box">
          <Mic size={20} strokeWidth={2.2} />
        </div>
        <div>
          <h1 className="brand-title">RAGawaz</h1>
          <div className="brand-meta">FAISS-HNSW • {embedderModel} • 5-Stage Guardrails</div>
        </div>
      </div>

      <div className="header-actions">
        {/* Language Selector: Defaults to Auto with manual overrides */}
        <div className="lang-selector">
          {LANGUAGES.map((lang) => (
            <button
              key={lang.code}
              className={`lang-btn ${selectedLanguage === lang.code ? 'active' : ''}`}
              onClick={() => setSelectedLanguage(lang.code)}
              title={lang.code === 'auto' ? 'Automatic language detection (Default)' : `Force input language to ${lang.label}`}
            >
              {lang.code === 'auto' && <Sparkles size={12} strokeWidth={2.5} />}
              <span>{lang.label}</span>
            </button>
          ))}
        </div>

        {/* Navigation Tabs */}
        <button
          className={`nav-tab-btn ${activeTab === 'voice' ? 'active' : ''}`}
          onClick={() => setActiveTab('voice')}
        >
          <Mic size={14} />
          <span>Voice Agent</span>
        </button>
        <button
          className={`nav-tab-btn ${activeTab === 'benchmark' ? 'active' : ''}`}
          onClick={() => setActiveTab('benchmark')}
        >
          <Activity size={14} />
          <span>Benchmarks</span>
        </button>

        {/* System Health Status */}
        <div className="health-badge">
          <span className="status-dot"></span>
          <Layers size={13} strokeWidth={2} />
          <span>{loadedCount.toLocaleString()} Chunks (14 Langs)</span>
        </div>
      </div>
    </header>
  );
}
