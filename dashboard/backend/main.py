"""FastAPI backend for the aero-opt dashboard."""
import sys, json, asyncio, threading, os
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # aero-opt root
sys.path.insert(0, str(Path(__file__).parent))                  # dashboard/backend

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

from rag_store import SimulationStore
from beam_stream import run_beam_optimization
from gear_stream import run_gear_optimization
from scipy_stream import run_scipy_beam
from interview import send_message as _interview_message
from fea_viz import simulate_fea_viz

_ROOT   = Path(__file__).parent.parent.parent   # aero-opt root
_DIST   = Path(__file__).parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Seed ChromaDB from bundled data on cold start (Render ephemeral disk)
    try:
        from startup_seed import seed_if_empty
        seed_if_empty()
    except Exception as exc:
        print(f"startup_seed skipped: {exc}")
    yield


app = FastAPI(title="Aero-Opt API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_beam_store = SimulationStore(path=str(_ROOT / "chroma_db"))
_gear_store = SimulationStore(path=str(_ROOT / "chroma_db_gear"))


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/stats")
def stats():
    return {
        "beam": {"run_count": _beam_store.run_count(), "lesson_count": _beam_store.lesson_count()},
        "gear": {"run_count": _gear_store.run_count(), "lesson_count": _gear_store.lesson_count()},
    }


@app.get("/api/runs")
def get_runs(domain: str = Query("beam"), limit: int = Query(50)):
    store = _beam_store if domain == "beam" else _gear_store
    if store.run_count() == 0:
        return {"runs": []}
    res  = store.runs.get(include=["documents", "metadatas"])
    runs = [
        {"id": rid, "document": doc, "metadata": meta}
        for rid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
    ]
    runs.sort(key=lambda r: r["metadata"].get("timestamp", ""), reverse=True)
    return {"runs": runs[:limit]}


@app.get("/api/best")
def get_best():
    def _best(store, mass_key):
        if store.run_count() == 0:
            return None
        res = store.runs.get(include=["metadatas"])
        candidates = [m for m in res["metadatas"] if m.get("feasible")]
        if not candidates:
            return None
        return min(candidates, key=lambda m: m.get(mass_key, float("inf")))

    return {
        "beam": _best(_beam_store, "mass_per_depth"),
        "gear": _best(_gear_store, "mass"),
    }


class BeamOptRequest(BaseModel):
    H_init: float = 0.006
    L: float = 0.100
    load: float = 500.0


class GearOptRequest(BaseModel):
    m_init: float = 2.0
    b_init: float = 25.0
    torque: float = 50.0


def _sse_stream(stream_fn, *args):
    async def generator():
        q    = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker():
            try:
                for event in stream_fn(*args):
                    loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception as exc:
                loop.call_soon_threadsafe(
                    q.put_nowait, {"type": "error", "message": str(exc)}
                )
            loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            event = await q.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return generator()


@app.post("/api/optimize/beam")
async def optimize_beam(body: BeamOptRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    params  = {"H_init": body.H_init, "L": body.L, "load": body.load}
    return StreamingResponse(
        _sse_stream(run_beam_optimization, params, api_key),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/optimize/gear")
async def optimize_gear(body: GearOptRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    params  = {"m_init": body.m_init, "b_init": body.b_init, "torque": body.torque}
    return StreamingResponse(
        _sse_stream(run_gear_optimization, params, api_key),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ScipyBeamRequest(BaseModel):
    H_init: float = 0.010
    L: float = 0.100
    load: float = 500.0


class InterviewRequest(BaseModel):
    messages: list


@app.post("/api/optimize/beam/scipy")
async def optimize_beam_scipy(body: ScipyBeamRequest):
    params = {"H_init": body.H_init, "L": body.L, "load": body.load}
    return StreamingResponse(
        _sse_stream(run_scipy_beam, params),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/interview/message")
async def interview_message(body: InterviewRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    result = await asyncio.to_thread(_interview_message, body.messages, api_key)
    return result


@app.get("/api/fea/viz")
def fea_viz(
    L:    float = Query(0.1),
    H:    float = Query(0.008308),
    load: float = Query(500.0),
):
    return simulate_fea_viz({"L": L, "H": H, "load": load})


# ── SPA static file serving (production) ─────────────────────────────────────
# Mounts AFTER all /api routes so FastAPI prefers exact matches.

if _DIST.exists():
    # Serve hashed asset bundles from /assets
    _assets = _DIST / "assets"
    if _assets.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="spa-assets")

    @app.get("/{full_path:path}")
    async def spa(full_path: str):
        """Catch-all: serve static file if it exists, otherwise return index.html."""
        candidate = _DIST / full_path
        if candidate.exists() and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(_DIST / "index.html"))
