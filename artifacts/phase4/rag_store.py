"""
Phase 3.5 -- Local RAG Store
==============================
ChromaDB-backed vector database for simulation runs and lessons learned.
Every simulate() call can be logged here; the agent queries it for context
before proposing new design parameters.

Usage
-----
    from rag_store import SimulationStore
    store = SimulationStore()
    store.log_run(params, result)
    hits = store.query_runs("beam height 8mm deflection near limit", n_results=3)
"""

import chromadb
from datetime import datetime

from simulate import _YIELD_PA

DELTA_MAX  = 0.050e-3
STRESS_MAX = 0.6 * _YIELD_PA
DB_PATH    = "./chroma_db"


class SimulationStore:
    """Persistent local vector store for simulation runs and lessons learned."""

    def __init__(self, path: str = DB_PATH):
        self.client = chromadb.PersistentClient(path=path)
        # Two collections: structured simulation runs + free-text lessons
        self.runs    = self.client.get_or_create_collection(name="simulation_runs")
        self.lessons = self.client.get_or_create_collection(name="lessons_learned")

    # ── Simulation runs ────────────────────────────────────────────────────────

    def log_run(self, params: dict, result: dict, note: str = "") -> str:
        """
        Store a simulation run as a natural-language document.
        Returns the run ID.
        """
        feasible = (
            result['max_deflection'] <= DELTA_MAX and
            result['max_von_mises']  <= STRESS_MAX
        )
        status = "FEASIBLE" if feasible else "INFEASIBLE"

        doc = (
            f"Beam design L={params.get('L', 0.1)*1e3:.1f}mm "
            f"H={params.get('H', 0.01)*1e3:.2f}mm "
            f"load={params.get('load', 500):.0f}N. "
            f"deflection={result['max_deflection']*1e3:.4f}mm "
            f"stress={result['max_von_mises']/1e6:.2f}MPa "
            f"mass={result['mass_per_depth']:.4f}kg_per_m. "
            f"Status={status}. {note}"
        )

        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.runs.add(
            documents=[doc],
            ids=[run_id],
            metadatas=[{
                "L":              params.get('L', 0.1),
                "H":              params.get('H', 0.01),
                "load":           params.get('load', 500.0),
                "max_deflection": result['max_deflection'],
                "max_von_mises":  result['max_von_mises'],
                "mass_per_depth": result['mass_per_depth'],
                "feasible":       feasible,
                "timestamp":      datetime.now().isoformat(),
            }]
        )
        return run_id

    def query_runs(self, query: str, n_results: int = 3) -> list[dict]:
        """Return the k most semantically similar past simulation runs."""
        n = min(n_results, self.runs.count())
        if n == 0:
            return []
        res = self.runs.query(query_texts=[query], n_results=n)
        return [
            {"document": doc, "metadata": meta}
            for doc, meta in zip(res['documents'][0], res['metadatas'][0])
        ]

    def run_count(self) -> int:
        return self.runs.count()

    # ── Lessons learned ────────────────────────────────────────────────────────

    def log_lesson(self, text: str, category: str = "general") -> str:
        """Store a free-text lesson or engineering note."""
        lesson_id = f"lesson_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.lessons.add(
            documents=[text],
            ids=[lesson_id],
            metadatas=[{"category": category, "timestamp": datetime.now().isoformat()}]
        )
        return lesson_id

    def query_lessons(self, query: str, n_results: int = 2) -> list[dict]:
        """Return the k most relevant lessons for a query."""
        n = min(n_results, self.lessons.count())
        if n == 0:
            return []
        res = self.lessons.query(query_texts=[query], n_results=n)
        return [
            {"document": doc, "metadata": meta}
            for doc, meta in zip(res['documents'][0], res['metadatas'][0])
        ]

    def lesson_count(self) -> int:
        return self.lessons.count()
