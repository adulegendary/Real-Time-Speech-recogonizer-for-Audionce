"""
Page-aware sliding-window semantic matcher for live liturgy tracking.

The liturgy is split into fixed-size pages (default 10 lines).
The window covers exactly the current page, with a small lookbehind
at page boundaries and early lookahead into the next page when we
approach the end.

Window behaviour
────────────────
  Page 1 (ids 1-10)   → window = [0, 9]
  Near end of page 1  → window = [0, 19]   (peek into page 2)
  Confirmed in page 2 → window = [8, 19]   (left shrinks, right = page 2 end)
  Near end of page 2  → window = [8, 29]   (peek into page 3)
  ...

Rules
─────
  1. confirmed_pos only moves FORWARD
  2. LEFT  : window_start = max(0, current_page_start - lookbehind)
  3. RIGHT : window_end   = current_page_end
  4. EXPAND: if confirmed_pos >= current_page_end - expand_threshold
             → window_end = next_page_end  (early lookahead)
"""

import json

import numpy as np


class LiturgyWindowMatcher:

    def __init__(
        self,
        vectors: np.ndarray,
        meta: list,
        page_size: int = 10,
        lookbehind: int = 2,
        expand_threshold: int = 2,
    ):
        """
        Parameters
        ----------
        vectors          : (N, 768) LaBSE embeddings, L2-normalised
        meta             : list of N dicts {id, role, section, text}
        page_size        : number of liturgy lines per page
        lookbehind       : lines back into previous page we still search
                           (helps at page boundaries where the priest may
                            repeat the closing line of the previous page)
        expand_threshold : when confirmed_pos is within this many lines of
                           the current page end, peek into the next page
        """
        if vectors.shape[0] != len(meta):
            raise ValueError("vectors and meta must have the same length")

        self.vectors = vectors.astype(np.float32)
        self.meta = meta
        self.N = len(meta)
        self.page_size = page_size
        self.lookbehind = lookbehind
        self.expand_threshold = expand_threshold

        # Build page boundaries: list of (start_idx, end_idx) inclusive
        self._pages: list[tuple[int, int]] = []
        for start in range(0, self.N, page_size):
            end = min(start + page_size - 1, self.N - 1)
            self._pages.append((start, end))

        self._current_page: int = 0
        self._confirmed_pos: int = 0
        self._window_start: int = 0
        self._window_end: int = self._pages[0][1]

    # ── public properties ──────────────────────────────────────────────────

    @property
    def confirmed_pos(self) -> int:
        return self._confirmed_pos

    @property
    def current_page(self) -> int:
        return self._current_page

    @property
    def window_start(self) -> int:
        return self._window_start

    @property
    def window_end(self) -> int:
        return self._window_end

    @property
    def window_size(self) -> int:
        return self._window_end - self._window_start + 1

    @property
    def total_pages(self) -> int:
        return len(self._pages)

    # ── matching ───────────────────────────────────────────────────────────

    def match_vector(self, query_vec: np.ndarray) -> dict:
        """
        Find the best liturgy line within the current page window.

        Parameters
        ----------
        query_vec : 1-D float32 array, L2-normalised (768-dim for LaBSE)

        Returns
        -------
        {
          "idx"          : int   — global index in the full liturgy,
          "score"        : float — cosine similarity [0, 1],
          "meta"         : dict  — {id, role, section, text},
          "page"         : int   — current page number (1-indexed for display),
          "window"       : (start, end),
          "window_size"  : int,
        }
        """
        q = query_vec.astype(np.float32).ravel()

        # Dot-product over only the current window slice
        window_vecs = self.vectors[self._window_start: self._window_end + 1]
        scores = window_vecs @ q

        local_idx = int(scores.argmax())
        best_score = float(scores[local_idx])
        global_idx = self._window_start + local_idx

        self._advance(global_idx)

        return {
            "idx": global_idx,
            "score": round(best_score, 4),
            "meta": self.meta[global_idx],
            "page": self._current_page + 1,
            "window": (self._window_start, self._window_end),
            "window_size": self.window_size,
        }

    def match_text(self, text: str, model) -> dict:
        """
        Encode `text` with the given SentenceTransformer model, then match.
        """
        vec = model.encode([text], normalize_embeddings=True)[0]
        return self.match_vector(vec)

    # ── window / page management ───────────────────────────────────────────

    def _advance(self, matched_pos: int):
        """Slide the page window forward after a match at `matched_pos`."""

        # Rule 1: confirmed position only moves forward
        if matched_pos > self._confirmed_pos:
            self._confirmed_pos = matched_pos

        # Find which page the confirmed position belongs to
        for page_idx, (pg_start, pg_end) in enumerate(self._pages):
            if pg_start <= self._confirmed_pos <= pg_end:
                self._current_page = page_idx
                break

        pg_start, pg_end = self._pages[self._current_page]

        # Rule 2: left shrinks to current page start minus small lookbehind
        self._window_start = max(0, pg_start - self.lookbehind)

        # Rule 3: right = end of current page
        self._window_end = pg_end

        # Rule 4: near-end expansion — peek into the next page early
        if self._confirmed_pos >= pg_end - self.expand_threshold:
            next_idx = self._current_page + 1
            if next_idx < len(self._pages):
                self._window_end = self._pages[next_idx][1]

    def reset(self):
        """Reset to the beginning of the liturgy."""
        self._current_page = 0
        self._confirmed_pos = 0
        self._window_start = 0
        self._window_end = self._pages[0][1]

    # ── inspection helpers ─────────────────────────────────────────────────

    def window_summary(self) -> str:
        """
        ASCII diagram showing the full liturgy, active window, and
        confirmed position.

        Example:
          [.........█---------..................]  pos=9  page=1/4  win=[7,19]  size=13/38
        """
        bar = ["."] * self.N
        for i in range(self._window_start, self._window_end + 1):
            bar[i] = "-"
        bar[self._confirmed_pos] = "█"
        return (
            f"[{''.join(bar)}]  "
            f"pos={self._confirmed_pos}  "
            f"page={self._current_page + 1}/{self.total_pages}  "
            f"win=[{self._window_start},{self._window_end}]  "
            f"size={self.window_size}/{self.N}"
        )

    def page_info(self) -> dict:
        """Return info about the current page."""
        pg_start, pg_end = self._pages[self._current_page]
        return {
            "page": self._current_page + 1,
            "total_pages": self.total_pages,
            "page_start": pg_start,
            "page_end": pg_end,
            "lines": self.meta[pg_start: pg_end + 1],
        }

    def current_line(self) -> dict:
        return self.meta[self._confirmed_pos]

    # ── factory ────────────────────────────────────────────────────────────

    @classmethod
    def from_files(
        cls,
        vectors_path: str = "data/embeddings/book_vectors.npy",
        meta_path: str = "data/embeddings/book_meta.json",
        **kwargs,
    ) -> "LiturgyWindowMatcher":
        """Load vectors and metadata from saved files."""
        vectors = np.load(vectors_path)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return cls(vectors, meta, **kwargs)


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    print("Loading LaBSE model...")
    model = SentenceTransformer("LaBSE")
    print("Model ready.\n")

    matcher = LiturgyWindowMatcher.from_files(page_size=10)

    # Print the page layout so we can see what each page covers
    print("=" * 60)
    print("  PAGE LAYOUT  (page_size=10)")
    print("=" * 60)
    for p in range(matcher.total_pages):
        pg_start, pg_end = matcher._pages[p]
        section_set = {matcher.meta[i]["section"] for i in range(pg_start, pg_end + 1)}
        print(f"  Page {p+1}: lines {pg_start+1}–{pg_end+1}  "
              f"(ids {matcher.meta[pg_start]['id']}–{matcher.meta[pg_end]['id']})  "
              f"sections: {', '.join(sorted(section_set))}")
    print()

    # Simulate phrases being spoken in order across pages
    test_queries = [
        # Page 1 (ids 1-10)
        ("priest",       "ቡረክ እግዚአብሔር አምላከ"),           # id 1
        ("congregation", "ወቡሩክ ስሙ"),                     # id 2
        ("priest",       "ሰላም ለኩሉ"),                     # id 4
        ("priest",       "ናስተሠናዕ ለእግዚአብሔር"),             # id 9  ← near page end
        ("congregation", "ዮቤ ወሰናዕ"),                     # id 10 ← page end, window should expand
        # Page 2 (ids 11-20)
        ("priest",       "ቅዱስ አንተ ወቅዱስ ስምከ"),            # id 11 ← flips to page 2
        ("congregation", "ቅዱስ ቅዱስ ቅዱስ እግዚአብሔር"),         # id 12
        ("priest",       "ዝንቱ ውእቱ ሥጋየ"),                 # id 16
    ]

    print(f"{'Speaker':<14} {'Query text':<32} {'→ Match':<30} {'Score':>5}  {'Page':>4}  Window")
    print("─" * 105)

    for speaker, query in test_queries:
        result = matcher.match_text(query, model)
        m = result["meta"]
        match_label = f"[id {m['id']}] {m['text'][:25]}"
        win = f"[{result['window'][0]},{result['window'][1]}] sz={result['window_size']}"
        print(
            f"{speaker:<14}  "
            f"{query:<32}  "
            f"{match_label:<30}  "
            f"{result['score']:>5.3f}  "
            f"p{result['page']:>1}/{matcher.total_pages}  "
            f"{win}"
        )
        print(f"  {matcher.window_summary()}\n")
