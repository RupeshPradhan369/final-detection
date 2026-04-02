const InputSection = ({ text, setText, onSubmit, loading }) => {
  return (
    <div style={{
      background: '#1e293b',
      borderRadius: '16px',
      padding: '32px',
      marginBottom: '24px',
      border: '1px solid #334155'
    }}>
      <h2 style={{ marginBottom: '16px', color: '#94a3b8', fontSize: '14px', textTransform: 'uppercase', letterSpacing: '1px' }}>
        Enter News Article
      </h2>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste a news article in English or Nepali..."
        style={{
          width: '100%',
          minHeight: '180px',
          background: '#0f172a',
          border: '1px solid #334155',
          borderRadius: '12px',
          padding: '16px',
          color: '#e2e8f0',
          fontSize: '15px',
          resize: 'vertical',
          outline: 'none',
          fontFamily: 'inherit',
          lineHeight: '1.6'
        }}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px' }}>
        <span style={{ color: '#64748b', fontSize: '13px' }}>
          {text.length} characters
        </span>
        <button
          onClick={onSubmit}
          disabled={loading || text.length < 20}
          style={{
            background: loading ? '#334155' : 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            color: 'white',
            border: 'none',
            borderRadius: '10px',
            padding: '12px 32px',
            fontSize: '15px',
            fontWeight: '600',
            cursor: loading || text.length < 20 ? 'not-allowed' : 'pointer',
            opacity: loading || text.length < 20 ? 0.6 : 1,
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