#!/usr/bin/env python3
"""
benchmark.py — Compare local LLMs and RAG embeddings on the beam optimization task.

LLM comparison  : llama3.1:8b vs qwen2.5:7b (Ollama)
  Metrics: convergence iterations, JSON parse failures, seconds/iter, final H accuracy

Embedding comparison: all-MiniLM-L6-v2 (ONNX) vs BAAI/bge-large-en-v1.5
  Metric: mean cosine distance for 5 domain-specific retrieval queries

Prerequisites:
  ollama serve                     (separate terminal)
  ollama pull llama3.1:8b
  ollama pull qwen2.5:7b
  pip install sentence-transformers  (for BGE)

Usage:
  python benchmark.py
"""

import json
import shutil
import time
import warnings
from pathlib import Path

import chromadb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

from simulate import simulate

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ─────────────────────────────────────────────────────────────────────
LLM_MODELS  = ["llama3.1:8b", "qwen2.5:7b"]
N_TRIALS    = 3
MAX_ITER    = 15
H0          = 6.0        # infeasible starting point
H_SCIPY     = 8.308      # scipy ground-truth optimum (mm)
DEF_LIM     = 0.05       # mm
STRESS_LIM  = 165.6      # MPa
CONV_TOL    = 0.005      # 0.5% convergence threshold
OLLAMA_URL  = "http://localhost:11434"

EMBED_QUERIES = [
    "infeasible deflection near 8mm height",
    "optimal mass reduction cantilever beam",
    "stress constraint violation high load",
    "converged feasible design near constraint boundary",
    "beam too flexible deflection exceeds limit",
]

# 12 representative documents seeded into each embedding collection
SEED_DOCS = [
    "H=6.0mm INFEASIBLE deflection=0.1023mm stress=104.2MPa mass=1.620kg/m — far too flexible",
    "H=7.0mm INFEASIBLE deflection=0.0640mm stress=76.5MPa mass=1.890kg/m — still over deflection limit",
    "H=8.0mm INFEASIBLE deflection=0.0443mm stress=58.5MPa mass=2.160kg/m — just over deflection limit",
    "H=8.3mm OK deflection=0.0396mm stress=54.4MPa mass=2.241kg/m — near optimal",
    "H=8.308mm OK deflection=0.0500mm stress=54.3MPa mass=2.243kg/m — scipy optimum, deflection binding",
    "H=9.0mm OK deflection=0.0313mm stress=44.4MPa mass=2.430kg/m — feasible but over-massed",
    "H=10.0mm OK deflection=0.0228mm stress=35.9MPa mass=2.700kg/m — initial design, too heavy",
    "H=12.0mm OK deflection=0.0132mm stress=24.9MPa mass=3.240kg/m — over-engineered",
    "Lesson: deflection scales as 1/H^3 — small reductions in H cause large deflection increases",
    "Lesson: constraint boundary near H=8.308mm — deflection constraint always binds first",
    "Lesson: increasing H by 10% reduces deflection by 27% due to cubic scaling",
    "Lesson: stress constraint never active — deflection always binds before stress limit is reached",
]

# ── Ollama helpers ─────────────────────────────────────────────────────────────

def ollama_ok() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).status_code == 200
    except Exception:
        return False

def model_installed(model: str) -> bool:
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).json().get("models", [])
        installed = {m["name"] for m in tags}
        bases = {n.split(":")[0] for n in installed}
        return model in installed or model.split(":")[0] in bases
    except Exception:
        return False

def ollama_generate(model: str, prompt: str) -> tuple[str, float]:
    t0 = time.time()
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.2}},
        timeout=300,
    )
    return r.json().get("response", ""), time.time() - t0

def parse_json_response(text: str) -> dict | None:
    """Extract JSON from LLM response, handling markdown fences."""
    clean = text.strip()
    # strip ```json ... ``` fences
    if "```" in clean:
        for part in clean.split("```"):
            part = part.strip().lstrip("json").strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue
    # direct parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    # find first {...} substring
    s, e = clean.find("{"), clean.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(clean[s : e + 1])
        except json.JSONDecodeError:
            pass
    return None

# ── LLM agent benchmark ────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "You are optimizing a cantilever aluminum beam. Goal: minimize mass (∝ H).\n"
    f"Constraints: deflection ≤ {DEF_LIM} mm, von Mises stress ≤ {STRESS_LIM} MPa.\n"
    "Physics: deflection ∝ 1/H³, stress ∝ 1/H². L = 100 mm fixed. H bounds [5, 15] mm.\n\n"
    "Current state:\n"
    "  H = {H:.3f} mm\n"
    "  deflection = {def_mm:.4f} mm  (limit {DEF_LIM} mm)\n"
    "  stress = {stress_mpa:.1f} MPa  (limit {STRESS_LIM} MPa)\n"
    "  mass = {mass:.4f} kg/m\n"
    "  Status: {status}\n\n"
    "Engineering context:\n{ctx}\n\n"
    "Reply with ONLY valid JSON: {{\"H_mm\": <float>, \"reasoning\": \"<one line>\"}}"
)

STATIC_CTX = (
    "- Deflection scales as 1/H^3: reducing H by 10% increases deflection by 37%.\n"
    "- Scipy optimum is H ≈ 8.308 mm where deflection constraint is binding.\n"
    "- If INFEASIBLE: increase H. If FEASIBLE and heavy: reduce H cautiously."
)

def run_llm_trial(model: str) -> dict:
    H = H0
    best_H: float | None = None
    best_mass = float("inf")
    json_fails = 0
    iter_times: list[float] = []
    H_hist = [H]

    for i in range(MAX_ITER):
        res = simulate({"L": 0.10, "H": H / 1000, "load": 500.0})
        def_mm = res["max_deflection"] * 1e3
        stress_mpa = res["max_von_mises"] / 1e6
        feasible = def_mm <= DEF_LIM and stress_mpa <= STRESS_LIM

        if feasible and res["mass_per_depth"] < best_mass:
            best_mass = res["mass_per_depth"]
            best_H = H

        # convergence: H barely moving and feasible solution exists
        if i >= 3 and best_H is not None and len(H_hist) >= 2:
            if abs(H_hist[-1] - H_hist[-2]) < 0.02:
                break

        prompt = PROMPT_TEMPLATE.format(
            H=H,
            def_mm=def_mm,
            DEF_LIM=DEF_LIM,
            stress_mpa=stress_mpa,
            STRESS_LIM=STRESS_LIM,
            mass=res["mass_per_depth"],
            status="INFEASIBLE" if not feasible else "FEASIBLE",
            ctx=STATIC_CTX,
        )

        response, elapsed = ollama_generate(model, prompt)
        iter_times.append(elapsed)

        parsed = parse_json_response(response)
        if parsed and "H_mm" in parsed:
            H = float(np.clip(parsed["H_mm"], 5.0, 15.0))
        else:
            json_fails += 1
            H = H * (1.10 if not feasible else 0.97)
            H = float(np.clip(H, 5.0, 15.0))

        H_hist.append(H)

    iterations = len(H_hist) - 1
    return {
        "model": model,
        "iterations": iterations,
        "json_failures": json_fails,
        "mean_time_s": float(np.mean(iter_times)) if iter_times else 0.0,
        "best_H": best_H,
        "error_pct": abs(best_H - H_SCIPY) / H_SCIPY * 100 if best_H else None,
        "H_hist": H_hist,
    }

# ── Embedding benchmark ────────────────────────────────────────────────────────

def _make_collection(db_path: str, ef=None) -> chromadb.Collection:
    """Create a fresh ChromaDB cosine collection seeded with SEED_DOCS."""
    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection("bench")
    except Exception:
        pass
    kwargs: dict = {"name": "bench", "metadata": {"hnsw:space": "cosine"}}
    if ef is not None:
        kwargs["embedding_function"] = ef
    col = client.create_collection(**kwargs)
    col.add(
        documents=SEED_DOCS,
        ids=[f"doc_{i}" for i in range(len(SEED_DOCS))],
    )
    return col

def _mean_dist(col: chromadb.Collection, query: str, n: int = 3) -> float:
    res = col.query(query_texts=[query], n_results=min(n, len(SEED_DOCS)))
    return float(np.mean(res["distances"][0]))

def run_embed_benchmark() -> dict[str, list[float] | None]:
    results: dict[str, list[float] | None] = {}

    print("  all-MiniLM-L6-v2 (ONNX, default)...")
    col_mini = _make_collection("./chroma_bench_mini")
    results["all-MiniLM-L6-v2"] = [_mean_dist(col_mini, q) for q in EMBED_QUERIES]

    print("  BAAI/bge-large-en-v1.5 (sentence-transformers)...")
    try:
        from sentence_transformers import SentenceTransformer

        class BGEFn(chromadb.EmbeddingFunction):
            def __init__(self):
                self._model = SentenceTransformer("BAAI/bge-large-en-v1.5")

            def __call__(self, input):  # noqa: A002
                return self._model.encode(
                    [f"Represent this sentence: {t}" for t in input],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()

        col_bge = _make_collection("./chroma_bench_bge", ef=BGEFn())
        results["BAAI/bge-large-en-v1.5"] = [_mean_dist(col_bge, q) for q in EMBED_QUERIES]
    except ImportError:
        print("    sentence-transformers not installed → pip install sentence-transformers")
        results["BAAI/bge-large-en-v1.5"] = None
    except Exception as exc:
        print(f"    BGE benchmark failed: {exc}")
        results["BAAI/bge-large-en-v1.5"] = None

    return results

# ── Plotting ───────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "llama3.1:8b":          "#4C72B0",
    "qwen2.5:7b":           "#DD8452",
    "all-MiniLM-L6-v2":    "#55A868",
    "BAAI/bge-large-en-v1.5": "#C44E52",
}

def plot_results(llm_results: dict, embed_results: dict, out: str = "benchmark_results.png"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Benchmark: Llama3.1:8b vs Qwen2.5:7b  |  MiniLM vs BGE-large Embeddings",
        fontsize=14, fontweight="bold", y=0.99,
    )

    # ── Panel A: H convergence curves ─────────────────────────────────────
    ax = axes[0, 0]
    ax.set_title("A — Beam Height Convergence (all trials)", fontweight="bold")
    for model, trials in llm_results.items():
        color = MODEL_COLORS.get(model, "gray")
        for trial in trials:
            ax.plot(range(len(trial["H_hist"])), trial["H_hist"],
                    color=color, alpha=0.35, lw=1.2)
        # mean trajectory (pad to max length)
        max_len = max(len(t["H_hist"]) for t in trials)
        padded = [t["H_hist"] + [t["H_hist"][-1]] * (max_len - len(t["H_hist"])) for t in trials]
        mean_traj = np.mean(padded, axis=0)
        ax.plot(range(len(mean_traj)), mean_traj, color=color, lw=2.5,
                label=model.split(":")[0].capitalize())
    ax.axhline(H_SCIPY, color="black", ls="--", lw=1.5, label=f"scipy optimum ({H_SCIPY} mm)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("H (mm)")
    ax.set_ylim(4, 16)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel B: LLM metric bars ───────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_title("B — LLM Agent Metrics (mean ± std, 3 trials)", fontweight="bold")
    metric_keys   = ["iterations", "json_failures", "mean_time_s", "error_pct"]
    metric_labels = ["Iterations\nto converge", "JSON\nfailures", "Sec /\niteration", "H error vs\nscipy (%)"]
    x = np.arange(len(metric_keys))
    width = 0.35
    for idx, (model, trials) in enumerate(llm_results.items()):
        means, stds = [], []
        for key in metric_keys:
            vals = [t[key] for t in trials if t[key] is not None]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        offset = (idx - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
               color=MODEL_COLORS.get(model, "gray"), alpha=0.85,
               label=model.split(":")[0].capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("Value")

    # ── Panel C: Embedding retrieval distances ─────────────────────────────
    ax = axes[1, 0]
    ax.set_title("C — RAG Retrieval Quality (cosine distance, lower = better match)",
                 fontweight="bold")
    short_q = [q[:28] + "…" if len(q) > 28 else q for q in EMBED_QUERIES]
    x = np.arange(len(EMBED_QUERIES))
    width = 0.35
    for idx, (name, scores) in enumerate(embed_results.items()):
        if scores is None:
            continue
        label = "MiniLM-L6-v2" if "MiniLM" in name else "BGE-large-en-v1.5"
        offset = (idx - 0.5) * width
        ax.bar(x + offset, scores, width, color=MODEL_COLORS.get(name, "gray"),
               alpha=0.85, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(short_q, fontsize=8, rotation=22, ha="right")
    ax.set_ylabel("Mean cosine distance  (↓ better)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel D: Summary table ─────────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("D — Summary", fontweight="bold")

    rows = [["Metric", "Llama3.1:8b", "Qwen2.5:7b"]]
    display = [
        ("iterations",    "Avg iterations"),
        ("json_failures", "JSON failures"),
        ("mean_time_s",   "Sec / iteration"),
        ("error_pct",     "H error vs scipy (%)"),
    ]
    for key, label in display:
        row = [label]
        for model in LLM_MODELS:
            if model in llm_results:
                vals = [t[key] for t in llm_results[model] if t[key] is not None]
                row.append(f"{np.mean(vals):.2f}" if vals else "—")
            else:
                row.append("not tested")
        rows.append(row)

    rows.append(["", "", ""])
    rows.append(["Embedding model", "MiniLM", "BGE-large"])

    mini = embed_results.get("all-MiniLM-L6-v2")
    bge  = embed_results.get("BAAI/bge-large-en-v1.5")
    rows.append([
        "Mean cosine dist (5 queries)",
        f"{np.mean(mini):.4f}" if mini else "—",
        f"{np.mean(bge):.4f}"  if bge  else "not tested",
    ])
    winner = "BGE" if (mini and bge and np.mean(bge) < np.mean(mini)) else "MiniLM"
    rows.append(["Better retrieval", "←" if winner == "MiniLM" else "", "←" if winner == "BGE" else ""])

    tbl = ax.table(
        cellText=[r for r in rows[1:]],
        colLabels=rows[0],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#DDDDDD")
            cell.set_text_props(fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  AERO-OPT BENCHMARK")
    print("  LLM: llama3.1:8b vs qwen2.5:7b")
    print("  Embeddings: all-MiniLM-L6-v2 vs BAAI/bge-large-en-v1.5")
    print("=" * 62)

    # ── LLM benchmark ─────────────────────────────────────────────────────
    llm_results: dict = {}

    if not ollama_ok():
        print("\n[LLM] Ollama not running — skipping LLM benchmark")
        print("      Start with: ollama serve")
    else:
        print(f"\n[LLM] Running {N_TRIALS} trials per model...")
        for model in LLM_MODELS:
            if not model_installed(model):
                print(f"  {model}: not installed — run: ollama pull {model}")
                continue
            print(f"\n  Model: {model}")
            trials = []
            for t_idx in range(N_TRIALS):
                print(f"    Trial {t_idx + 1}/{N_TRIALS}...", end=" ", flush=True)
                trial = run_llm_trial(model)
                trials.append(trial)
                err = f"{trial['error_pct']:.2f}%" if trial["error_pct"] is not None else "no feasible"
                print(
                    f"{trial['iterations']} iters | "
                    f"{trial['json_failures']} JSON fails | "
                    f"{trial['mean_time_s']:.1f}s/iter | "
                    f"H error {err}"
                )
            llm_results[model] = trials

    # ── Embedding benchmark ────────────────────────────────────────────────
    print("\n[Embeddings] Seeding and querying collections...")
    embed_results = run_embed_benchmark()

    for name, scores in embed_results.items():
        tag = "MiniLM" if "MiniLM" in name else "BGE-large"
        if scores:
            print(f"  {tag}: mean cosine dist = {np.mean(scores):.4f}  "
                  f"(per query: {[f'{s:.3f}' for s in scores]})")
        else:
            print(f"  {tag}: skipped")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)

    for model, trials in llm_results.items():
        def mean_std(key):
            vals = [t[key] for t in trials if t[key] is not None]
            return (np.mean(vals), np.std(vals)) if vals else (0, 0)

        it_m, it_s = mean_std("iterations")
        jf_m, jf_s = mean_std("json_failures")
        ti_m, ti_s = mean_std("mean_time_s")
        er_m, er_s = mean_std("error_pct")
        print(f"\n{model}:")
        print(f"  Iterations to converge : {it_m:.1f} ± {it_s:.1f}")
        print(f"  JSON parse failures    : {jf_m:.1f} ± {jf_s:.1f}")
        print(f"  Seconds per iteration  : {ti_m:.2f} ± {ti_s:.2f}")
        print(f"  Final H error vs scipy : {er_m:.3f}% ± {er_s:.3f}%")

    mini = embed_results.get("all-MiniLM-L6-v2")
    bge  = embed_results.get("BAAI/bge-large-en-v1.5")
    print("\nEmbedding retrieval (mean cosine dist, lower = better):")
    print(f"  all-MiniLM-L6-v2        : {np.mean(mini):.4f}" if mini else "  all-MiniLM-L6-v2 : —")
    print(f"  BAAI/bge-large-en-v1.5  : {np.mean(bge):.4f}"  if bge  else "  BAAI/bge-large-en-v1.5 : not tested")
    if mini and bge:
        winner = "BGE-large-en-v1.5" if np.mean(bge) < np.mean(mini) else "all-MiniLM-L6-v2"
        improvement = abs(np.mean(mini) - np.mean(bge)) / np.mean(mini) * 100
        print(f"  Winner: {winner} ({improvement:.1f}% lower distance)")

    # ── Plot ──────────────────────────────────────────────────────────────
    if llm_results or any(v for v in embed_results.values()):
        plot_results(llm_results, embed_results, "benchmark_results.png")
        Path("artifacts/benchmark").mkdir(parents=True, exist_ok=True)
        shutil.copy("benchmark_results.png", "artifacts/benchmark/benchmark_results.png")
        print("Copied → artifacts/benchmark/benchmark_results.png")

    # cleanup temp ChromaDB directories
    for p in ["./chroma_bench_mini", "./chroma_bench_bge"]:
        shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main()
