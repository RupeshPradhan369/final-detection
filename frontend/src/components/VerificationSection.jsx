const VerificationSection = ({
  apiVerification,
  rssVerification,
  keywords,
}) => {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: "24px",
        marginBottom: "24px",
      }}
    >
      {/* Keywords */}
      <div
        style={{
          background: "#ffffff",
          borderRadius: "16px",
          padding: "24px",
          border: "1px solid #e2e8f0",
          boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
        }}
      >
        <h3
          style={{
            color: "#475569",
            fontSize: "12px",
            textTransform: "uppercase",
            letterSpacing: "1px",
            marginBottom: "16px",
            fontWeight: "600",
          }}
        >
          Extracted Keywords
        </h3>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
          {keywords.map((kw, i) => (
            <span
              key={i}
              style={{
                background: "#ede9fe",
                color: "#5b21b6",
                padding: "6px 14px",
                borderRadius: "20px",
                fontSize: "13px",
                fontWeight: "500",
                border: "1px solid #c4b5fd",
              }}
            >
              {kw}
            </span>
          ))}
        </div>
      </div>

      {/* RSS Coverage */}
      <div
        style={{
          background: "#ffffff",
          borderRadius: "16px",
          padding: "24px",
          border: "1px solid #e2e8f0",
          boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
        }}
      >
        <h3
          style={{
            color: "#475569",
            fontSize: "12px",
            textTransform: "uppercase",
            letterSpacing: "1px",
            marginBottom: "16px",
            fontWeight: "600",
          }}
        >
          RSS Coverage — {rssVerification.coverage}
        </h3>
        {rssVerification.matched_sources.length > 0 ? (
          rssVerification.matched_sources.map((src, i) => (
            <div
              key={i}
              style={{
                background: "#f0fdf4",
                border: "1px solid #bbf7d0",
                borderRadius: "8px",
                padding: "10px 14px",
                marginBottom: "10px",
              }}
            >
              <div
                style={{
                  color: "#15803d",
                  fontSize: "13px",
                  fontWeight: "600",
                  marginBottom: "4px",
                }}
              >
                {src.source}
              </div>
              <a
                href={src.link}
                target="_blank"
                rel="noreferrer"
                style={{
                  color: "#64748b",
                  fontSize: "12px",
                  textDecoration: "none",
                }}
              >
                {src.title}
              </a>
            </div>
          ))
        ) : (
          <div style={{ textAlign: "center", padding: "16px 0" }}>
            <div style={{ fontSize: "24px", marginBottom: "8px" }}>📭</div>
            <p style={{ color: "#94a3b8", fontSize: "13px" }}>
              No matching sources found
            </p>
            <p style={{ color: "#cbd5e1", fontSize: "12px", marginTop: "4px" }}>
              This story may not be covered by monitored feeds
            </p>
          </div>
        )}
      </div>

      {/* Fact Check API */}
      <div
        style={{
          background: "#ffffff",
          borderRadius: "16px",
          padding: "24px",
          border: "1px solid #e2e8f0",
          boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
          gridColumn: "1 / -1",
        }}
      >
        <h3
          style={{
            color: "#475569",
            fontSize: "12px",
            textTransform: "uppercase",
            letterSpacing: "1px",
            marginBottom: "16px",
            fontWeight: "600",
          }}
        >
          Fact Check API Results
        </h3>
        {apiVerification.found ? (
          apiVerification.results.map((result, i) => (
            <div
              key={i}
              style={{
                background: "#fffbeb",
                borderRadius: "10px",
                padding: "16px",
                marginBottom: "12px",
                border: "1px solid #fde68a",
              }}
            >
              <div
                style={{
                  color: "#1e293b",
                  fontSize: "14px",
                  marginBottom: "10px",
                  lineHeight: "1.5",
                }}
              >
                {result.claim}
              </div>
              <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
                <span
                  style={{
                    background: "#f1f5f9",
                    color: "#475569",
                    fontSize: "12px",
                    padding: "3px 10px",
                    borderRadius: "6px",
                    fontWeight: "500",
                  }}
                >
                  📰 {result.publisher}
                </span>
                <span
                  style={{
                    background: "#fef3c7",
                    color: "#92400e",
                    fontSize: "12px",
                    padding: "3px 10px",
                    borderRadius: "6px",
                    fontWeight: "600",
                  }}
                >
                  ⚖️ {result.rating}
                </span>
              </div>
            </div>
          ))
        ) : (
          <div style={{ textAlign: "center", padding: "16px 0" }}>
            <div style={{ fontSize: "24px", marginBottom: "8px" }}>🔍</div>
            <p style={{ color: "#94a3b8", fontSize: "13px" }}>
              No fact-check records found for this article.
            </p>
            <p style={{ color: "#cbd5e1", fontSize: "12px", marginTop: "4px" }}>
              This does not mean the article is credible — check RSS coverage
              above.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default VerificationSection;
