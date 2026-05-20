const LimeExplanation = ({ explanation }) => {
  if (!explanation || explanation.length === 0) return null;
  const maxWeight = Math.max(...explanation.map((e) => Math.abs(e.weight)));

  return (
    <div
      style={{
        background: "#ffffff",
        borderRadius: "16px",
        padding: "24px",
        marginBottom: "24px",
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
          marginBottom: "6px",
          fontWeight: "600",
        }}
      >
        LIME Explanation — Word Importance
      </h3>
      <p style={{ color: "#94a3b8", fontSize: "12px", marginBottom: "20px" }}>
        Each word's contribution to the classification decision
      </p>

      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        {explanation.map((item, i) => {
          const isFake = item.weight > 0;
          const barWidth = (Math.abs(item.weight) / maxWeight) * 100;

          return (
            <div
              key={i}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                background: isFake ? "#fff5f5" : "#f0fdf4",
                borderRadius: "8px",
                padding: "8px 12px",
                border: `1px solid ${isFake ? "#fecaca" : "#bbf7d0"}`,
              }}
            >
              {/* Word */}
              <span
                style={{
                  width: "130px",
                  textAlign: "right",
                  color: "#1e293b",
                  fontSize: "13px",
                  fontWeight: "600",
                  flexShrink: 0,
                }}
              >
                {item.word}
              </span>

              {/* Bar */}
              <div
                style={{
                  flex: 1,
                  background: "#e2e8f0",
                  borderRadius: "4px",
                  height: "18px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${barWidth}%`,
                    height: "100%",
                    background: isFake
                      ? "linear-gradient(90deg, #ef4444, #dc2626)"
                      : "linear-gradient(90deg, #22c55e, #16a34a)",
                    borderRadius: "4px",
                    transition: "width 0.5s ease",
                  }}
                />
              </div>

              {/* Weight */}
              <span
                style={{
                  width: "56px",
                  color: isFake ? "#dc2626" : "#16a34a",
                  fontSize: "12px",
                  fontWeight: "700",
                  flexShrink: 0,
                }}
              >
                {item.weight > 0 ? "+" : ""}
                {item.weight}
              </span>

              {/* Badge */}
              <span
                style={{
                  background: isFake ? "#dc2626" : "#16a34a",
                  color: "white",
                  fontSize: "10px",
                  fontWeight: "700",
                  padding: "2px 8px",
                  borderRadius: "4px",
                  flexShrink: 0,
                  letterSpacing: "0.5px",
                }}
              >
                {isFake ? "FAKE" : "REAL"}
              </span>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div
        style={{
          marginTop: "20px",
          padding: "12px 16px",
          background: "#f8fafc",
          borderRadius: "8px",
          border: "1px solid #e2e8f0",
          display: "flex",
          gap: "24px",
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div
            style={{
              width: "16px",
              height: "10px",
              background: "#ef4444",
              borderRadius: "2px",
            }}
          />
          <span style={{ color: "#475569", fontSize: "12px" }}>
            Pushes toward FAKE
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div
            style={{
              width: "16px",
              height: "10px",
              background: "#22c55e",
              borderRadius: "2px",
            }}
          />
          <span style={{ color: "#475569", fontSize: "12px" }}>
            Pushes toward REAL
          </span>
        </div>
        <span style={{ color: "#94a3b8", fontSize: "12px" }}>
          Longer bar = stronger influence on the decision
        </span>
      </div>
    </div>
  );
};

export default LimeExplanation;
