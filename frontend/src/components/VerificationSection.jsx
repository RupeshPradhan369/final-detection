const VerificationSection = ({ apiVerification, rssVerification, keywords }) => {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '24px' }}>
      
      {/* Keywords */}
      <div style={{ background: '#1e293b', borderRadius: '16px', padding: '24px', border: '1px solid #334155' }}>
        <h3 style={{ color: '#94a3b8', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '16px' }}>
          Extracted Keywords
        </h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
          {keywords.map((kw, i) => (
            <span key={i} style={{
              background: '#312e81',
              color: '#a5b4fc',
              padding: '6px 14px',
              borderRadius: '20px',
              fontSize: '13px',
              border: '1px solid #4338ca'
            }}>
              {kw}
            </span>
          ))}
        </div>
      </div>

      {/* RSS Coverage */}
      <div style={{ background: '#1e293b', borderRadius: '16px', padding: '24px', border: '1px solid #334155' }}>
        <h3 style={{ color: '#94a3b8', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '16px' }}>
          RSS Coverage — {rssVerification.coverage}
        </h3>
        {rssVerification.matched_sources.length > 0 ? (
          rssVerification.matched_sources.map((src, i) => (
            <div key={i} style={{ marginBottom: '12px' }}>
              <div style={{ color: '#22c55e', fontSize: '13px', fontWeight: '600' }}>{src.source}</div>
              <a href={src.link} target="_blank" rel="noreferrer"
                style={{ color: '#64748b', fontSize: '12px', textDecoration: 'none' }}>
                {src.title}
              </a>
            </div>
          ))
        ) : (
          <p style={{ color: '#64748b', fontSize: '13px' }}>No matching sources found</p>
        )}
      </div>

      {/* Fact Check API */}
      <div style={{ background: '#1e293b', borderRadius: '16px', padding: '24px', border: '1px solid #334155', gridColumn: '1 / -1' }}>
        <h3 style={{ color: '#94a3b8', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '16px' }}>
          Fact Check API Results
        </h3>
        {apiVerification.found ? (
          apiVerification.results.map((result, i) => (
            <div key={i} style={{
              background: '#0f172a',
              borderRadius: '10px',
              padding: '16px',
              marginBottom: '12px',
              border: '1px solid #334155'
            }}>
              <div style={{ color: '#e2e8f0', fontSize: '14px', marginBottom: '8px' }}>{result.claim}</div>
              <div style={{ display: 'flex', gap: '16px' }}>
                <span style={{ color: '#94a3b8', fontSize: '12px' }}>📰 {result.publisher}</span>
                <span style={{ color: '#f59e0b', fontSize: '12px' }}>⚖️ {result.rating}</span>
              </div>
            </div>
          ))
        ) : (
          <p style={{ color: '#64748b', fontSize: '13px' }}>
            No fact-check records found for this article. This does not mean the article is credible.
          </p>
        )}
      </div>
    </div>
  );
};

export default VerificationSection;