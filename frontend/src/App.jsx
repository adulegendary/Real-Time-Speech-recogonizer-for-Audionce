import { useEffect, useRef } from "react";
import { useLiturgySocket } from "./useLiturgySocket";
import liturgy from "./liturgy.json";
import "./App.css";

// Where the backend WebSocket lives. The fake server (or your real pipeline)
// listens here.
const WS_URL = "ws://localhost:8000/ws";

// Small helper: turn a 0..1 score into a color (red -> yellow -> green).
function scoreColor(v) {
  const hue = Math.round(120 * Math.max(0, Math.min(1, v))); // 0=red, 120=green
  return `hsl(${hue}, 70%, 45%)`;
}

function ConfidenceBar({ label, value }) {
  const pct = Math.round((value ?? 0) * 100);
  return (
    <div className="bar-row">
      <span className="bar-label">{label}</span>
      <div className="bar-track">
        <div
          className="bar-fill"
          style={{ width: `${pct}%`, background: scoreColor(value ?? 0) }}
        />
      </div>
      <span className="bar-pct">{pct}%</span>
    </div>
  );
}

export default function App() {
  // The ONE line that wires Python -> React. `match` updates on every message.
  const { match, status } = useLiturgySocket(WS_URL);

  // Ref to the active <li> so we can auto-scroll it into view.
  const activeRef = useRef(null);
  useEffect(() => {
    activeRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [match?.id]); // runs whenever the matched line id changes

  return (
    <div className="app">
      <header className="topbar">
        <h1>Liturgy — Live Match</h1>
        <span className={`status status--${status}`}>
          {status === "open" ? "🟢 Live" : status === "connecting" ? "🟡 Connecting" : "🔴 Offline"}
        </span>
      </header>

      {/* What the backend last heard + how confident it is */}
      <section className="now">
        {match ? (
          <>
            <div className="heard">
              heard: <em>“{match.transcript}”</em>{" "}
              <span className="backend">via {match.backend}</span>
            </div>
            <ConfidenceBar label="match" value={match.score} />
            <ConfidenceBar label="stt" value={match.confidence} />
          </>
        ) : (
          <div className="heard">Waiting for the first match…</div>
        )}
      </section>

      {/* The full liturgy; the active line is highlighted */}
      <ol className="liturgy">
        {liturgy.map((line) => {
          const isActive = match && line.id === match.id;
          return (
            <li
              key={line.id}
              ref={isActive ? activeRef : null}
              className={`line ${isActive ? "line--active" : ""}`}
            >
              <span className={`role role--${line.role}`}>{line.role}</span>
              <span className="text">{line.text}</span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
