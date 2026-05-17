"""
Phase 5 -- STL Export for 3D Printing
=======================================
Generates a printable STL of the optimal cantilever beam geometry
using gmsh (OpenCASCADE kernel for clean boolean operations).

Geometry
--------
  Beam   : L x H_opt x depth  (optimal cross-section from Phase 3/4)
  Tab    : 20mm mounting block at the clamped end (for vise/C-clamp)
  Hole   : 4mm dia at tip (weight hook for load test)

Usage
-----
  python export_stl.py                          # scipy optimal (Al-6061 basis)
  python export_stl.py --H 8.32                 # agent optimal
  python export_stl.py --depth 30               # thicker print
  python export_stl.py --out my_beam.stl

Output
------
  cantilever_optimal.stl  --  ready to slice and print
"""

import gmsh
import argparse
import os

# ── CLI ───────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Phase 5 -- Export optimal beam to STL")
_parser.add_argument("--H",     type=float, default=8.308,
                     help="Beam height mm  (default 8.308 = scipy Phase 3 optimal)")
_parser.add_argument("--L",     type=float, default=100.0,
                     help="Beam length mm  (default 100)")
_parser.add_argument("--depth", type=float, default=20.0,
                     help="Print depth (thickness) mm  (default 20)")
_parser.add_argument("--out",   type=str,   default="cantilever_optimal.stl",
                     help="Output STL filename")
ARGS = _parser.parse_args()

H_mm    = ARGS.H
L_mm    = ARGS.L
depth   = ARGS.depth
out     = ARGS.out

# ── Derived geometry ──────────────────────────────────────────────────────────
TAB_L  = 20.0        # mounting tab length (extends behind beam, -x direction)
TAB_H  = 20.0        # mounting tab total height (centered on beam centroid)
HOLE_D = 4.0         # tip hole diameter (mm)

beam_cy = H_mm / 2.0
tab_y0  = beam_cy - TAB_H / 2.0   # tab bottom (may be below y=0)
hole_cx = L_mm - 6.0               # hole center x: 6mm setback from tip
hole_cy = beam_cy                  # hole center y: beam midheight
hole_r  = HOLE_D / 2.0


def _eb_deflection(H_mm, L_mm, depth_mm, P_N, E_Pa):
    """Euler-Bernoulli tip deflection for rectangular solid beam (mm)."""
    H = H_mm * 1e-3
    L = L_mm * 1e-3
    b = depth_mm * 1e-3
    I = b * H**3 / 12.0
    return P_N * L**3 / (3.0 * E_Pa * I) * 1e3   # → mm


# ── Banner ────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Phase 5 -- STL Export")
print(f"{'='*55}")
print(f"  Beam       : L={L_mm:.0f} mm  H={H_mm:.3f} mm  depth={depth:.0f} mm")
print(f"  Tab        : {TAB_L:.0f} mm x {TAB_H:.0f} mm (clamped-end mounting block)")
print(f"  Tip hole   : dia={HOLE_D:.0f} mm at x={hole_cx:.0f} mm (weight hook)")

# ── Geometry (OCC kernel) ─────────────────────────────────────────────────────
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)   # suppress gmsh log spam
gmsh.model.add("cantilever")

beam_s = gmsh.model.occ.addRectangle(0,      0,      0, L_mm,  H_mm)
tab_s  = gmsh.model.occ.addRectangle(-TAB_L, tab_y0, 0, TAB_L, TAB_H)
hole_s = gmsh.model.occ.addDisk(hole_cx, hole_cy, 0, hole_r, hole_r)

# Fuse beam + tab into one body, then punch the tip hole
fused,  _ = gmsh.model.occ.fuse([(2, beam_s)], [(2, tab_s)])
profile, _ = gmsh.model.occ.cut(fused, [(2, hole_s)])

# Extrude 2D profile → 3D solid
gmsh.model.occ.extrude(profile, 0, 0, depth)
gmsh.model.occ.synchronize()

# ── Mesh settings (fine enough for print accuracy, not for FEA) ───────────────
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.0)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.25)
gmsh.option.setNumber("Mesh.Algorithm3D", 4)   # Frontal-Delaunay

gmsh.model.mesh.generate(3)
gmsh.write(out)
gmsh.finalize()

size_kb = os.path.getsize(out) / 1024

# ── Post-export info ──────────────────────────────────────────────────────────
print(f"\n  Written: {out}  ({size_kb:.0f} KB)")

print(f"\n  --- Expected tip deflection at design load (P=500 N) ---")
d_al  = _eb_deflection(H_mm, L_mm, depth, 500.0, 70.0e9)
d_pla = _eb_deflection(H_mm, L_mm, depth, 500.0,  3.5e9)
print(f"  Al-6061 (E=70 GPa) : {d_al:.3f} mm")
print(f"  PLA     (E= 3.5 GPa): {d_pla:.3f} mm")
print(f"  (Ratio {d_pla/d_al:.0f}x -- printed part deflects far more than the aluminum design)")

print(f"\n  --- PLA load-deflection table (depth={depth:.0f} mm, H={H_mm:.3f} mm) ---")
print(f"  {'Load (N)':>10}  {'Load (kg)':>10}  {'Deflection (mm)':>16}")
for P in [1, 2, 5, 10, 20, 50]:
    d = _eb_deflection(H_mm, L_mm, depth, float(P), 3.5e9)
    print(f"  {P:>10}  {P/9.81:>10.2f}  {d:>16.3f}")
print(f"  (calipers read to 0.01 mm -- loads of 2 N and above are clearly measurable)")

print(f"""
  --- Slicer settings ---
  Material    : PLA (proof of concept, E~3.5 GPa) or PETG (E~2.1 GPa)
  Layer height: 0.15 mm  (dimensional accuracy over speed)
  Infill      : 100%     (solid cross-section required for FEA match)
  Supports    : none     (no overhangs in this geometry)
  Orientation : beam flat on bed, tab end at back, hole facing up

  --- Load test procedure ---
  1. Clamp the {TAB_L:.0f} mm tab in a vise or C-clamp (flush with table edge)
  2. Zero calipers at the tip before loading
  3. Hang calibrated weights from a wire through the {HOLE_D:.0f} mm tip hole
  4. Record (load_N, deflection_mm) pairs -- aim for 5-8 data points
  5. Save as measured_data.csv  (header: load_N,deflection_mm)
  6. Run: python validate_physical.py --material pla

  --- Notes ---
  * The geometry is Al-6061 optimal (H={H_mm:.3f} mm reduces mass 16.9% vs baseline).
    The print is PLA -- material mismatch is intentional and documented.
    validate_physical.py compares measurements to PLA FEA, not Al FEA.
  * 100% infill is critical -- FEA assumes solid material, voids invalidate comparison.
  * Check L and H of the printed part with calipers before testing.
    Record actual measured dimensions and pass them to validate_physical.py.
""")
print(f"{'='*55}\n")
