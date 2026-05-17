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

    for domain, db_name, mass_key in [
        ("beam", "chroma_db",      "mass_per_depth"),
        ("gear", "chroma_db_gear", "mass"),
    ]:
        try:
            store = SimulationStore(path=str(_ROOT / db_name))
            if store.run_count() == 0:
                runs = data.get(domain, [])
                for i, run in enumerate(runs):
                    meta = {k: v for k, v in run["metadata"].items() if v is not None}
                    store.runs.add(
                        documents=[run["document"]],
                        ids=[f"seed_{domain}_{i}"],
                        metadatas=[meta],
                    )
                print(f"startup_seed: seeded {len(runs)} {domain} runs")
            else:
                print(f"startup_seed: {domain} store has {store.run_count()} runs, skipping")
        except Exception as exc:
            print(f"startup_seed: {domain} seeding failed ({exc}) — continuing without seed")
