const VerdictCard = ({ verdict, unifiedScore, prediction }) => {
  const isFake = verdict === "FAKE";

  return (
    <div
      style={{
        background: isFake
          ? "linear-gradient(135deg, #7f1d1d, #991b1b)"
          : "linear-gradient(135deg, #14532d, #166534)",
        borderRadius: "16px",
        padding: "32px",
        marginBottom: "24px",
        border: `1px solid ${isFake ? "#ef4444" : "#22c55e"}`,
        textAlign: "center",
        boxShadow: "0 4px 16px rgba(0,0,0,0.2)",
      }}
    >
      {/* Icon */}
      <div style={{ fontSize: "48px", marginBottom: "8px" }}>
        {isFake ? "⚠️" : "✅"}
      </div>

      {/* REAL / FAKE label */}
      <h1
        style={{
          fontSize: "48px",
          fontWeight: "800",
          color: "#ffffff",
          marginBottom: "12px",
          letterSpacing: "2px",
        }}
      >
        {verdict}
      </h1>

      {/* Unified Score */}
      <p
        style={{
          color: "#cbd5e1",
          fontSize: "16px",
          marginBottom: "28px",
        }}
      >
        Unified Credibility Score:{" "}
        <strong
          style={{
            color: isFake ? "#fca5a5" : "#86efac",
            fontSize: "18px",
          }}
        >
          {unifiedScore}%
        </strong>
      </p>

      {/* Stats — Fake % and Real % only */}
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          gap: "48px",
        }}
      >
        <div>
          <div
            style={{
              color: "#ef4444",
              fontSize: "28px",
              fontWeight: "700",
            }}
          >
            {prediction.fake_probability}%
          </div>
          <div style={{ color: "#94a3b8", fontSize: "13px", marginTop: "4px" }}>
            Fake Probability
          </div>
        </div>

        <div>
          <div
            style={{
              color: "#22c55e",
              fontSize: "28px",
              fontWeight: "700",
            }}
          >
            {prediction.real_probability}%
          </div>
          <div style={{ color: "#94a3b8", fontSize: "13px", marginTop: "4px" }}>
            Real Probability
          </div>
        </div>
      </div>
    </div>
  );
};

export default VerdictCard;
