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
from chromadb import EmbeddingFunction
from datetime import datetime

DB_PATH = "./chroma_db"

# Beam-specific constants (used only for beam log_run document generation)
try:
    from simulate import _YIELD_PA
    _DELTA_MAX  = 0.050e-3
    _STRESS_MAX = 0.6 * _YIELD_PA
except ImportError:
    _DELTA_MAX  = 5e-5
    _STRESS_MAX = 165.6e6


class _BGEEmbeddingFn(EmbeddingFunction):
    """BAAI/bge-large-en-v1.5 via sentence-transformers — 47.9% better retrieval than MiniLM."""
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    def __call__(self, input):  # noqa: A002
        return self._model.encode(
            [f"Represent this sentence: {t}" for t in input],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()


def _get_embedding_fn():
    """Return BGE embedding function, falling back to ChromaDB default if unavailable."""
    try:
        return _BGEEmbeddingFn()
    except ImportError:
        return None  # ChromaDB falls back to all-MiniLM-L6-v2 via ONNX


class SimulationStore:
    """Persistent local vector store for simulation runs and lessons learned."""

    def __init__(self, path: str = DB_PATH, db_path: str = None, embedding_fn=None):
        # accept both 'path' and 'db_path' keyword for convenience
        resolved = db_path if db_path is not None else path
        self.client = chromadb.PersistentClient(path=resolved)
        ef = embedding_fn if embedding_fn is not None else _get_embedding_fn()
        col_kwargs = {"embedding_function": ef} if ef is not None else {}
        self.runs    = self.client.get_or_create_collection(name="simulation_runs",    **col_kwargs)
        self.lessons = self.client.get_or_create_collection(name="lessons_learned",   **col_kwargs)

    # ── Simulation runs ────────────────────────────────────────────────────────

    def log_run(self, params: dict, result: dict, note: str = "") -> str:
        """
        Store a simulation run as a natural-language document.
        Handles both beam results (max_deflection / max_von_mises) and
        gear results (bending_stress / contact_stress) automatically.
        Returns the run ID.
        """
        # Build a generic document from whatever result keys are present
        if 'bending_stress' in result:
            # Gear result
            feasible = (result['bending_stress'] <= 180.0 and
                        result['contact_stress']  <= 550.0)
            status = "FEASIBLE" if feasible else "INFEASIBLE"
            doc = (
                f"Gear design m={params.get('m',3):.2f}mm "
                f"b={params.get('b',40):.1f}mm "
                f"torque={params.get('torque',50):.0f}Nm. "
                f"bending={result['bending_stress']:.1f}MPa "
                f"contact={result['contact_stress']:.1f}MPa "
                f"mass={result['mass']*1000:.1f}g "
                f"pitch_diam={result.get('pitch_diam_mm',0):.1f}mm. "
                f"Status={status}. {note}"
            )
            meta = {
                "m": float(params.get('m', 0)),
                "b": float(params.get('b', 0)),
                "bending_stress": float(result['bending_stress']),
                "contact_stress": float(result['contact_stress']),
                "mass":           float(result['mass']),
                "feasible":       feasible,
                "timestamp":      datetime.now().isoformat(),
            }
        else:
            # Beam result (original)
            feasible = (result['max_deflection'] <= _DELTA_MAX and
                        result['max_von_mises']  <= _STRESS_MAX)
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
            meta = {
                "L":              params.get('L', 0.1),
                "H":              params.get('H', 0.01),
                "load":           params.get('load', 500.0),
                "max_deflection": result['max_deflection'],
                "max_von_mises":  result['max_von_mises'],
                "mass_per_depth": result['mass_per_depth'],
                "feasible":       feasible,
                "timestamp":      datetime.now().isoformat(),
            }

        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.runs.add(documents=[doc], ids=[run_id], metadatas=[meta])
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
