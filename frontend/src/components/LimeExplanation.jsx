const LimeExplanation = ({ explanation }) => {
  const maxWeight = Math.max(...explanation.map(e => Math.abs(e.weight)));

  return (
    <div style={{
      background: '#1e293b',
      borderRadius: '16px',
      padding: '24px',
      marginBottom: '24px',
      border: '1px solid #334155'
    }}>
      <h3 style={{
        color: '#94a3b8',
        fontSize: '12px',
        textTransform: 'uppercase',
        letterSpacing: '1px',
        marginBottom: '8px'
      }}>
        LIME Explanation — Word Importance
      </h3>
      <p style={{ color: '#64748b', fontSize: '12px', marginBottom: '20px' }}>
        🔴 Red words push toward FAKE — 🟢 Green words push toward REAL
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {explanation.map((item, i) => {
          const isFake = item.weight > 0;
          const barWidth = Math.abs(item.weight) / maxWeight * 100;
          return (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{
                width: '140px',
                textAlign: 'right',
                color: '#e2e8f0',
                fontSize: '13px',
                fontWeight: '500'
              }}>
                {item.word}
              </span>
              <div style={{
                flex: 1,
                background: '#0f172a',
                borderRadius: '4px',
                height: '20px'
              }}>
                <div style={{
                  width: `${barWidth}%`,
                  height: '100%',
                  background: isFake ? '#ef4444' : '#22c55e',
                  borderRadius: '4px',
                  transition: 'width 0.5s ease'
                }} />
              </div>
              <span style={{
                width: '60px',
                color: isFake ? '#ef4444' : '#22c55e',
                fontSize: '12px',
                fontWeight: '600'
              }}>
                {item.weight > 0 ? '+' : ''}{item.weight}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default LimeExplanation;