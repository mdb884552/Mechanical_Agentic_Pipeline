"""Stream scipy SLSQP beam optimization — yields same event schema as beam_stream.py.

Uses thread + queue so scipy callback events flow out in real time rather than
all at once when minimize() returns.
"""
import sys, queue, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scipy.optimize import minimize
from simulate import simulate, _YIELD_PA

DELTA_MAX  = 0.050e-3
STRESS_MAX = 0.6 * _YIELD_PA
H_MIN, H_MAX = 0.003, 0.030
RHO = 2700.0


def run_scipy_beam(params: dict):
    """Generator: yields JSON-serialisable event dicts matching beam_stream format.

    No 'reasoning' events are emitted — that absence is the visual contrast with
    the agent in the Race page.
    """
    L  = float(params.get("L",      0.100))
    P  = float(params.get("load",   500.0))
    H0 = float(params.get("H_init", 0.010))

    q      = queue.Queue()
    cache  = {}
    iter_n = [0]

    def _sim(H):
        if H not in cache:
            cache[H] = simulate({"L": L, "H": H, "load": P, "ny": 6})
        return cache[H]

    def objective(x):
        return RHO * L * x[0]

    def c_deflection(x):
        return DELTA_MAX - _sim(x[0])["max_deflection"]

    def c_stress(x):
        return STRESS_MAX - _sim(x[0])["max_von_mises"]

    def callback(x):
        r = _sim(x[0])
        feasible = (r["max_deflection"] <= DELTA_MAX and
                    r["max_von_mises"]  <= STRESS_MAX)
        iter_n[0] += 1
        q.put({
            "type":          "iteration",
            "iter":          iter_n[0],
            "H_mm":          round(x[0] * 1e3, 3),
            "deflection_mm": round(r["max_deflection"] * 1e3, 4),
            "stress_mpa":    round(r["max_von_mises"] / 1e6, 1),
            "mass":          round(r["mass_per_depth"], 4),
            "feasible":      feasible,
        })

    def worker():
        try:
            result = minimize(
                objective,
                x0=[H0],
                method="SLSQP",
                bounds=[(H_MIN, H_MAX)],
                constraints=[
                    {"type": "ineq", "fun": c_deflection},
                    {"type": "ineq", "fun": c_stress},
                ],
                callback=callback,
                options={"ftol": 1e-10, "maxiter": 200},
            )
            r_opt = _sim(result.x[0])
            q.put({
                "type":                 "done",
                "best_H_mm":            round(result.x[0] * 1e3, 3),
                "best_mass":            round(r_opt["mass_per_depth"], 4),
                "iterations":           iter_n[0],
                "scipy_baseline_H_mm":  8.308,
                "scipy_baseline_mass":  2.2432,
            })
        except Exception as exc:
            q.put({"type": "error", "message": str(exc)})
        q.put(None)

    yield {"type": "start", "L_mm": round(L * 1e3, 1), "load_N": P, "H_init_mm": round(H0 * 1e3, 2)}

    threading.Thread(target=worker, daemon=True).start()

    while True:
        ev = q.get()
        if ev is None:
            break
        yield ev
