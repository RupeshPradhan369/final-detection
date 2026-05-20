const InputSection = ({ text, setText, onSubmit, loading }) => {
  const isReady = text.length >= 20;

  return (
    <div style={{
      background: '#ffffff',
      borderRadius: '16px',
      padding: '32px',
      marginBottom: '24px',
      border: '1px solid #e2e8f0',
      boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
    }}>
      <h2 style={{
        marginBottom: '16px',
        color: '#475569',
        fontSize: '13px',
        textTransform: 'uppercase',
        letterSpacing: '1px',
        fontWeight: '600'
      }}>
        Enter News Article
      </h2>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste a news article in English or Nepali..."
        style={{
          width: '100%',
          minHeight: '180px',
          background: '#f8fafc',
          border: '1.5px solid #e2e8f0',
          borderRadius: '12px',
          padding: '16px',
          color: '#1e293b',
          fontSize: '15px',
          resize: 'vertical',
          outline: 'none',
          fontFamily: 'inherit',
          lineHeight: '1.6',
          transition: 'border-color 0.2s',
          boxSizing: 'border-box'
        }}
        onFocus={e => e.target.style.borderColor = '#6366f1'}
        onBlur={e => e.target.style.borderColor = '#e2e8f0'}
      />

      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginTop: '16px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ color: '#94a3b8', fontSize: '13px' }}>
            {text.length} characters
          </span>
          {isReady && (
            <span style={{
              background: '#dcfce7',
              color: '#16a34a',
              fontSize: '12px',
              fontWeight: '600',
              padding: '2px 10px',
              borderRadius: '20px'
            }}>
              ✓ Ready
            </span>
          )}
        </div>

        <button
          onClick={onSubmit}
          disabled={loading || !isReady}
          style={{
            background: loading || !isReady
              ? '#e2e8f0'
              : 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            color: loading || !isReady ? '#94a3b8' : 'white',
            border: 'none',
            borderRadius: '10px',
            padding: '12px 32px',
            fontSize: '15px',
            fontWeight: '600',
            cursor: loading || !isReady ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s'
          }}
        >
          {loading ? 'Analyzing...' : 'Analyze Article'}
        </button>
      </div>
    </div>
  );
};

export default InputSection;