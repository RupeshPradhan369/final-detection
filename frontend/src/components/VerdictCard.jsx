const VerdictCard = ({ verdict, unifiedScore, prediction }) => {
  const isFake = verdict === 'FAKE';

  return (
    <div style={{
      background: isFake
        ? 'linear-gradient(135deg, #7f1d1d, #991b1b)'
        : 'linear-gradient(135deg, #14532d, #166534)',
      borderRadius: '16px',
      padding: '32px',
      marginBottom: '24px',
      border: `1px solid ${isFake ? '#ef4444' : '#22c55e'}`,
      textAlign: 'center'
    }}>
      <div style={{ fontSize: '48px', marginBottom: '8px' }}>
        {isFake ? '⚠️' : '✅'}
      </div>
      <h1 style={{
        fontSize: '48px',
        fontWeight: '800',
        color: isFake ? '#fca5a5' : '#86efac',
        marginBottom: '8px'
      }}>
        {verdict}
      </h1>
      <p style={{ color: '#cbd5e1', fontSize: '16px', marginBottom: '24px' }}>
        Unified Credibility Score: <strong style={{ color: isFake ? '#fca5a5' : '#86efac' }}>
          {unifiedScore}%
        </strong>
      </p>
      <div style={{ display: 'flex', justifyContent: 'center', gap: '32px' }}>
        <div>
          <div style={{ color: '#ef4444', fontSize: '24px', fontWeight: '700' }}>
            {prediction.fake_probability}%
          </div>
          <div style={{ color: '#94a3b8', fontSize: '12px' }}>Fake Probability</div>
        </div>
        <div>
          <div style={{ color: '#22c55e', fontSize: '24px', fontWeight: '700' }}>
            {prediction.real_probability}%
          </div>
          <div style={{ color: '#94a3b8', fontSize: '12px' }}>Real Probability</div>
        </div>
        <div>
          <div style={{ color: '#a78bfa', fontSize: '24px', fontWeight: '700' }}>
            {prediction.confidence}%
          </div>
          <div style={{ color: '#94a3b8', fontSize: '12px' }}>Model Confidence</div>
        </div>
      </div>
    </div>
  );
};

export default VerdictCard;