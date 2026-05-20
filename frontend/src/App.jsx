import { useState } from 'react';
import { predictNews, explainNews } from './api';
import InputSection from './components/InputSection';
import VerdictCard from './components/VerdictCard';
import VerificationSection from './components/VerificationSection';
import LimeExplanation from './components/LimeExplanation';

const LOADING_MESSAGES = [
  'Analyzing article with XLM-RoBERTa...',
  'Checking 29 RSS feeds in parallel...',
  'Querying Google Fact Check API...',
  'Computing unified credibility score...',
];

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState(0);
  const [error, setError] = useState('');
  const [explanation, setExplanation] = useState(null);
  const [explainLoading, setExplainLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    setExplanation(null);
    setLoadingMsg(0);

    // Cycle through loading messages
    const interval = setInterval(() => {
      setLoadingMsg(prev => (prev + 1) % LOADING_MESSAGES.length);
    }, 1800);

    try {
      const data = await predictNews(text);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.error || 'Something went wrong. Is the backend running?');
    } finally {
      clearInterval(interval);
      setLoading(false);
    }
  };

  const handleExplain = async () => {
    setExplainLoading(true);
    try {
      const data = await explainNews(text);
      setExplanation(data.explanation);
    } catch (err) {
      setError('LIME explanation failed: ' + (err.response?.data?.error || err.message));
    } finally {
      setExplainLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f1f5f9', padding: '40px 20px' }}>
      <div style={{ maxWidth: '900px', margin: '0 auto' }}>

        {/* Campus Badge */}
        <div style={{ textAlign: 'center', marginBottom: '12px' }}>
          <span style={{
            background: '#e0e7ff',
            color: '#3730a3',
            fontSize: '12px',
            fontWeight: '600',
            padding: '4px 14px',
            borderRadius: '20px',
            letterSpacing: '0.5px'
          }}>
            Mount Annapurna Campus — B.Sc. CSIT Final Year Project 2026
          </span>
        </div>

        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '16px' }}>
          <h1 style={{
            fontSize: '36px',
            fontWeight: '800',
            background: 'linear-gradient(135deg, #4f46e5, #7c3aed, #db2777)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            marginBottom: '16px'
          }}>
            Fake News Detection System
          </h1>

          {/* Tech Badges */}
          <div style={{ display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: '8px', marginBottom: '8px' }}>
            {['XLM-RoBERTa', 'Django REST', 'React', 'LIME', 'RSS Verification'].map(badge => (
              <span key={badge} style={{
                background: '#1e293b',
                color: '#94a3b8',
                fontSize: '11px',
                fontWeight: '600',
                padding: '4px 10px',
                borderRadius: '6px',
                letterSpacing: '0.3px'
              }}>
                {badge}
              </span>
            ))}
          </div>
        </div>

        {/* Input */}
        <InputSection
          text={text}
          setText={setText}
          onSubmit={handleSubmit}
          loading={loading}
        />

        {/* Error */}
        {error && (
          <div style={{
            background: '#fee2e2',
            border: '1px solid #fca5a5',
            borderRadius: '12px',
            padding: '16px',
            marginBottom: '24px',
            color: '#991b1b',
            fontWeight: '500'
          }}>
            ⚠️ {error}
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div style={{ textAlign: 'center', padding: '48px', color: '#64748b' }}>
            {/* Spinner */}
            <div style={{
              width: '48px',
              height: '48px',
              border: '4px solid #e2e8f0',
              borderTop: '4px solid #6366f1',
              borderRadius: '50%',
              margin: '0 auto 20px',
              animation: 'spin 1s linear infinite'
            }} />
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            <p style={{ fontWeight: '600', color: '#1e293b', fontSize: '15px', marginBottom: '6px' }}>
              {LOADING_MESSAGES[loadingMsg]}
            </p>
            <p style={{ fontSize: '13px', color: '#94a3b8' }}>
              This may take 5–10 seconds
            </p>
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <>
            <VerdictCard
              verdict={result.verdict}
              unifiedScore={result.unified_score}
              prediction={result.prediction}
            />
            <VerificationSection
              apiVerification={result.api_verification}
              rssVerification={result.rss_verification}
              keywords={result.keywords}
            />

            {/* LIME Button */}
            <div style={{ textAlign: 'center', marginBottom: '24px' }}>
              <button
                onClick={handleExplain}
                disabled={explainLoading}
                style={{
                  background: explainLoading
                    ? '#e2e8f0'
                    : 'linear-gradient(135deg, #0891b2, #0e7490)',
                  color: explainLoading ? '#94a3b8' : 'white',
                  border: 'none',
                  borderRadius: '10px',
                  padding: '12px 32px',
                  fontSize: '15px',
                  fontWeight: '600',
                  cursor: explainLoading ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                {explainLoading ? '⏳ Generating Explanation...' : '🔬 Explain with LIME'}
              </button>
              <p style={{ color: '#94a3b8', fontSize: '12px', marginTop: '8px' }}>
                Takes ~30 seconds — shows which words influenced the decision
              </p>
            </div>

            {/* LIME Results */}
            {explanation && <LimeExplanation explanation={explanation} />}
          </>
        )}

        {/* Footer */}
        <div style={{
          textAlign: 'center',
          marginTop: '40px',
          color: '#94a3b8',
          fontSize: '13px'
        }}>
          <div style={{ marginBottom: '4px' }}>
            Powered by XLM-RoBERTa • Django REST Framework • React
          </div>
          <div style={{ color: '#cbd5e1', fontSize: '12px' }}>
            Pawan Gurung • Rupesh Pradhan • Bhesh Bahadur Saru
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;