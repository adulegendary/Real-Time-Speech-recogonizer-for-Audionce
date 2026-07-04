import { useEffect, useMemo, useRef, useState } from "react";
import { useLiturgySocket } from "./useLiturgySocket";
import liturgy from "./liturgy.json";
import "./App.css";

// Where the backend WebSocket lives. The fake server (or your real pipeline)
// listens here.
const WS_URL = "ws://localhost:8000/ws";

// How many liturgy lines show on one "page" of the book. Change this to make
// pages taller/shorter for the projector/screen you're using.
const PAGE_SIZE = 5;

// Length of the page-flip animation, kept in sync with the CSS keyframes.
const FLIP_MS = 600;

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

  // Split the flat liturgy into fixed-size pages once (never changes).
  const pages = useMemo(() => {
    const out = [];
    for (let i = 0; i < liturgy.length; i += PAGE_SIZE) {
      out.push(liturgy.slice(i, i + PAGE_SIZE));
    }
    return out;
  }, []);

  // `page` = the page currently shown. `flip` drives the CSS turn animation
  // ("fwd" | "back" | null). Guard prevents starting a new turn mid-flip.
  const [page, setPage] = useState(0);
  const [flip, setFlip] = useState(null);
  const flipping = useRef(false);

  function turnTo(target) {
    if (flipping.current || target === page || target < 0 || target >= pages.length) return;
    const dir = target > page ? "fwd" : "back";
    flipping.current = true;
    setFlip(dir);
    // Swap the page content at the midpoint (page is edge-on, so the swap is
    // hidden), then clear the animation class when it finishes.
    setTimeout(() => setPage(target), FLIP_MS / 2);
    setTimeout(() => {
      setFlip(null);
      flipping.current = false;
    }, FLIP_MS);
  }

  // Auto-flip: when the matched line lands on a different page, turn to it.
  useEffect(() => {
    if (!match) return;
    const idx = liturgy.findIndex((l) => l.id === match.id);
    if (idx < 0) return;
    turnTo(Math.floor(idx / PAGE_SIZE));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [match?.id]);

  const lines = pages[page] ?? [];

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

      {/* The liturgy as a flip-book: one page at a time, active line lit up */}
      <div className="book" style={{ perspective: "1800px" }}>
        <ol className={`page ${flip ? `page--flip-${flip}` : ""}`}>
          {lines.map((line) => {
            const isActive = match && line.id === match.id;
            return (
              <li key={line.id} className={`line ${isActive ? "line--active" : ""}`}>
                <span className={`role role--${line.role}`}>{line.role}</span>
                <span className="text">{line.text}</span>
              </li>
            );
          })}
        </ol>
      </div>

      {/* Page controls — for rehearsal or when the stream is offline */}
      <nav className="pager">
        <button className="pager-btn" onClick={() => turnTo(page - 1)} disabled={page === 0}>
          ◀
        </button>
        <div className="pager-dots">
          {pages.map((_, i) => (
            <span
              key={i}
              className={`dot ${i === page ? "dot--on" : ""}`}
              onClick={() => turnTo(i)}
            />
          ))}
        </div>
        <button
          className="pager-btn"
          onClick={() => turnTo(page + 1)}
          disabled={page === pages.length - 1}
        >
          ▶
        </button>
        <span className="pager-count">
          {page + 1} / {pages.length}
        </span>
      </nav>
    </div>
  );
}
