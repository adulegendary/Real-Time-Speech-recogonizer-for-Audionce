"""
Sliding-window semantic matcher for live liturgy tracking.

The window keeps only a small slice of the liturgy in play at any time:

  Full liturgy:  [0  1  2  3  4  5  6  7  8  9 ... 37]
                          |←  active window  →|
                    start                   end
                               ↑
                          confirmed_pos

Rules:
  - LEFT shrinks:  window_start = confirmed_pos - lookbehind
    (we discard lines we've already passed; small lookbehind handles repetitions)
  - RIGHT expands: when confirmed_pos is within `expand_threshold` lines of
    window_end, push window_end forward by another `lookahead` chunk
  - confirmed_pos only moves FORWARD — the liturgy never goes backward
"""

import json
import os

import numpy as np


class LiturgyWindowMatcher:

    def __init__(
        self,
        vectors: np.ndarray,
        meta: list,
        lookahead: int = 6,
        lookbehind: int = 2,
        expand_threshold: int = 2,
    ):
        """
        Parameters
        ----------
        vectors          : (N, 768) LaBSE embeddings, L2-normalised
        meta             : list of N dicts {id, role, section, text}
        lookahead        : lines to search ahead of confirmed position
        lookbehind       : lines back we still allow (handles short repetitions)
        expand_threshold : expand right when match is this close to window_end
        """
        if vectors.shape[0] != len(meta):
            raise ValueError("vectors and meta must have the same length")

        self.vectors = vectors.astype(np.float32)
        self.meta = meta
        self.N = len(meta)
        self.lookahead = lookahead
        self.lookbehind = lookbehind
        self.expand_threshold = expand_threshold

        self._confirmed_pos: int = 0
        self._window_start: int = 0
        self._window_end: int = min(lookahead, self.N - 1)

    # ── public properties ──────────────────────────────────────────────────

    @property
    def confirmed_pos(self) -> int:
        return self._confirmed_pos

    @property
    def window_start(self) -> int:
        return self._window_start

    @property
    def window_end(self) -> int:
        return self._window_end

    @property
    def window_size(self) -> int:
        return self._window_end - self._window_start + 1

    # ── matching ───────────────────────────────────────────────────────────

    def match_vector(self, query_vec: np.ndarray) -> dict:
        """
        Find the best liturgy line within the current window.

        Parameters
        ----------
        query_vec : 1-D float32 array, L2-normalised (768-dim for LaBSE)

        Returns
        -------
        {
          "idx"         : int    — global index in the full liturgy,
          "score"       : float  — cosine similarity [0, 1],
          "meta"        : dict   — {id, role, section, text},
          "window"      : (start, end),
          "window_size" : int,
        }
        """
        q = query_vec.astype(np.float32).ravel()

        # Slice the window and compute cosine similarities in one dot-product
        window_vecs = self.vectors[self._window_start: self._window_end + 1]
        scores = window_vecs @ q  # shape (window_size,)

        local_idx = int(scores.argmax())
        best_score = float(scores[local_idx])
        global_idx = self._window_start + local_idx

        self._advance(global_idx)

        return {
            "idx": global_idx,
            "score": round(best_score, 4),
            "meta": self.meta[global_idx],
            "window": (self._window_start, self._window_end),
            "window_size": self.window_size,
        }

    def match_text(self, text: str, model) -> dict:
        """
        Encode `text` with the given SentenceTransformer model, then match.
        `model` should be a SentenceTransformer instance (LaBSE recommended).
        """
        vec = model.encode([text], normalize_embeddings=True)[0]
        return self.match_vector(vec)

    # ── window management ──────────────────────────────────────────────────

    def _advance(self, matched_pos: int):
        """Slide the window forward after a match at `matched_pos`."""

        # Rule 1: confirmed position only moves forward
        if matched_pos > self._confirmed_pos:
            self._confirmed_pos = matched_pos

        # Rule 2: left shrinks — drop dead lines behind lookbehind
        self._window_start = max(0, self._confirmed_pos - self.lookbehind)

        # Rule 3: right expands from confirmed position
        desired_end = self._confirmed_pos + self.lookahead

        # Rule 4: near-end expansion — if we're close to the current right
        #         boundary, push it forward an extra chunk so we never run
        #         out of lookahead before the window updates
        if self._confirmed_pos >= self._window_end - self.expand_threshold:
            desired_end += self.lookahead

        self._window_end = min(self.N - 1, desired_end)

    def reset(self):
        """Reset to the beginning of the liturgy."""
        self._confirmed_pos = 0
        self._window_start = 0
        self._window_end = min(self.lookahead, self.N - 1)

    # ── debug helpers ──────────────────────────────────────────────────────

    def window_summary(self) -> str:
        """One-line ASCII diagram of the current window state."""
        total = self.N
        bar = list("." * total)
        for i in range(self._window_start, self._window_end + 1):
            bar[i] = "-"
        bar[self._confirmed_pos] = "█"
        return (
            f"[{''.join(bar)}]  "
            f"pos={self._confirmed_pos}  "
            f"win=[{self._window_start},{self._window_end}]  "
            f"size={self.window_size}/{total}"
        )

    def current_line(self) -> dict:
        """Return metadata for the current confirmed position."""
        return self.meta[self._confirmed_pos]

    # ── factory ────────────────────────────────────────────────────────────

    @classmethod
    def from_files(
        cls,
        vectors_path: str = "data/embeddings/book_vectors.npy",
        meta_path: str = "data/embeddings/book_meta.json",
        **kwargs,
    ) -> "LiturgyWindowMatcher":
        """Load from saved numpy + json files."""
        vectors = np.load(vectors_path)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return cls(vectors, meta, **kwargs)


# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    print("Loading LaBSE model...")
    model = SentenceTransformer("LaBSE")
    print("Model ready.\n")

    matcher = LiturgyWindowMatcher.from_files()

    # Simulate walking through the liturgy with a few spoken phrases
    test_queries = [
        ("priest", "ቡረክ እግዚአብሔር"),           # Opening  → line 0
        ("congregation", "ወቡሩክ ስሙ"),           # Opening  → line 1
        ("deacon", "ቁሙ ለጸሎት"),                # Opening  → line 2
        ("priest", "ሰላም ለኩሉ"),                 # Greeting → line 3
        ("congregation", "ወለመንፈስከ"),           # Greeting → line 4
        ("priest", "አሳዕሎ ልቦናክሙ"),              # Sursum   → line 6
        ("congregation", "ቅዱስ ቅዱስ ቅዱስ"),      # Sanctus  → line 11
        ("priest", "አቡነ ዘበሰማያት"),              # Lord's Prayer → line 22
        ("congregation", "አሜን አሜን አሜን"),        # Blessing → line 37
    ]

    print(f"{'Speaker':<14} {'Query':<30} {'Match':<35} {'Score':>6}  {'Window'}")
    print("─" * 110)

    for speaker, query in test_queries:
        result = matcher.match_text(query, model)
        m = result["meta"]
        win = f"[{result['window'][0]},{result['window'][1]}] sz={result['window_size']}"
        print(
            f"{speaker:<14} "
            f"{query:<30} "
            f"[{m['role']}] {m['text'][:32]:<35} "
            f"{result['score']:>6.3f}  "
            f"{win}"
        )
        print(f"  {matcher.window_summary()}")
        print()
