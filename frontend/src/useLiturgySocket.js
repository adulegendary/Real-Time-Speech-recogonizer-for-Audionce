import { useState, useEffect, useRef } from "react";

/**
 * Connects to the backend WebSocket and exposes the latest liturgy match.
 *
 * @param {string} url  e.g. "ws://localhost:8000/ws"
 * @returns {{ match: object|null, status: "connecting"|"open"|"closed" }}
 */
export function useLiturgySocket(url) {
  // `match` = the latest payload from Python. Changing it re-renders the UI.
  const [match, setMatch] = useState(null);
  // `status` drives the "🟢 Live / 🔴 Disconnected" indicator.
  const [status, setStatus] = useState("connecting");
  // The live WebSocket object. A ref (not state) because it's machinery,
  // not display data — changing it must NOT trigger a re-render.
  const socketRef = useRef(null);

  useEffect(() => {
    let ws;
    let reconnectTimer;
    let closedByUs = false; // guard: don't reconnect during intentional teardown

    function connect() {
      setStatus("connecting");
      ws = new WebSocket(url); // browser-native — no library needed
      socketRef.current = ws;

      ws.onopen = () => setStatus("open");

      // Fires every time Python does ws.send(json.dumps(payload)).
      ws.onmessage = (event) => {
        const payload = JSON.parse(event.data); // JSON string -> JS object
        setMatch(payload);
      };

      ws.onclose = () => {
        setStatus("closed");
        // Native WebSocket does NOT auto-reconnect — we do it ourselves.
        if (!closedByUs) {
          reconnectTimer = setTimeout(connect, 1500);
        }
      };

      // On error, force a close so onclose runs and schedules a retry.
      ws.onerror = () => ws.close();
    }

    connect();

    // Cleanup: runs when the component unmounts or `url` changes.
    return () => {
      closedByUs = true; // stop the reconnect loop
      clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, [url]); // re-run only if the target URL changes

  return { match, status };
}
