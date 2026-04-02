import { useState } from 'react';
import { predictNews, explainNews } from './api';
import InputSection from './components/InputSection';
import VerdictCard from './components/VerdictCard';
import VerificationSection from './components/VerificationSection';
import LimeExplanation from './components/LimeExplanation';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [explanation, setExplanation] = useState(null);
  const [explainLoading, setExplainLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    setExplanation(null);
    try {
      const data = await predictNews(text);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.error || 'Something went wrong. Is the backend running?');
    } finally {
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
    <div style={{ minHeight: '100vh', background: '#0f172a', padding: '40px 20px' }}>
      <div style={{ maxWidth: '900px', margin: '0 auto' }}>

        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '40px' }}>
          <h1 style={{
            fontSize: '36px',
            fontWeight: '800',
            background: 'linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            marginBottom: '8px'
          }}>
            Fake News Detection System
          </h1>
          <p style={{ color: '#64748b', fontSize: '15px' }}>
            Multilingual AI-powered news verification — English & Nepali
          </p>
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
            background: '#7f1d1d',
            border: '1px solid #ef4444',
            borderRadius: '12px',
            padding: '16px',
            marginBottom: '24px',
            color: '#fca5a5'
          }}>
            ⚠️ {error}
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>
            <div style={{ fontSize: '32px', marginBottom: '16px' }}>🔍</div>
            <p>Analyzing article with XLM-RoBERTa...</p>
            <p style={{ fontSize: '13px', marginTop: '8px' }}>
              Checking RSS feeds and fact-check APIs...
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
                    ? '#334155'
                    : 'linear-gradient(135deg, #0891b2, #0e7490)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '10px',
                  padding: '12px 32px',
                  fontSize: '15px',
                  fontWeight: '600',
                  cursor: explainLoading ? 'not-allowed' : 'pointer',
                  opacity: explainLoading ? 0.6 : 1
                }}
              >
                {explainLoading ? '⏳ Generating Explanation...' : '🔬 Explain with LIME'}
              </button>
              <p style={{ color: '#64748b', fontSize: '12px', marginTop: '8px' }}>
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
          color: '#334155',
          fontSize: '13px'
        }}>
          Powered by XLM-RoBERTa • Django REST Framework • React
        </div>
      </div>
    </div>
  );
}

export default App;