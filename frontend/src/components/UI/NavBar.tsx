import React from "react";

/**
 * Static header — no tabs (no replay mode in the new frontend).
 */
export const NavBar: React.FC = () => {
  return (
    <nav style={styles.navbar}>
      <div style={styles.container}>
        <div style={styles.title}>Quoridor</div>
      </div>
    </nav>
  );
};

const styles = {
  navbar: {
    width: "100%",
    backgroundColor: "#ffffff",
    borderBottom: "1px solid #e5e7eb",
    boxShadow: "0 1px 3px rgba(0, 0, 0, 0.1)",
    position: "fixed" as const,
    top: 0,
    left: 0,
    zIndex: 1000,
  },
  container: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "12px 24px",
    maxWidth: "100%",
    margin: "0 auto",
  },
  title: {
    fontSize: "24px",
    fontWeight: 700,
    color: "#1f2937",
    letterSpacing: "-0.5px",
  },
};
