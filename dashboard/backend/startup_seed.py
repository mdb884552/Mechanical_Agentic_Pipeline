"""
Seed ChromaDB stores from seed_data.json on cold start.
Only runs if the store is empty — safe to call on every startup.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # aero-opt root
sys.path.insert(0, str(Path(__file__).parent))                  # dashboard/backend

from rag_store import SimulationStore

_SEED_FILE = Path(__file__).parent / "seed_data.json"
_ROOT      = Path(__file__).parent.parent.parent


def seed_if_empty():
    if not _SEED_FILE.exists():
        print("startup_seed: seed_data.json not found, skipping")
        return

    data = json.loads(_SEED_FILE.read_text(encoding="utf-8"))

    beam_store = SimulationStore(path=str(_ROOT / "chroma_db"))
    if beam_store.run_count() == 0:
        runs = data.get("beam", [])
        for i, run in enumerate(runs):
            meta = {k: v for k, v in run["metadata"].items() if v is not None}
            beam_store.runs.add(
                documents=[run["document"]],
                ids=[f"seed_beam_{i}"],
                metadatas=[meta],
            )
        print(f"startup_seed: seeded {len(runs)} beam runs")
    else:
        print(f"startup_seed: beam store has {beam_store.run_count()} runs, skipping")

    gear_store = SimulationStore(path=str(_ROOT / "chroma_db_gear"))
    if gear_store.run_count() == 0:
        runs = data.get("gear", [])
        for i, run in enumerate(runs):
            meta = {k: v for k, v in run["metadata"].items() if v is not None}
            gear_store.runs.add(
                documents=[run["document"]],
                ids=[f"seed_gear_{i}"],
                metadatas=[meta],
            )
        print(f"startup_seed: seeded {len(runs)} gear runs")
    else:
        print(f"startup_seed: gear store has {gear_store.run_count()} runs, skipping")
