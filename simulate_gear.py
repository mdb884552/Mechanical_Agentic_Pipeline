"""
Gear Simulator -- Phase analogue of simulate.py
================================================
Analytical gear stress calculation using:
  - Lewis equation for tooth bending (AGMA basis)
  - Hertz contact stress (AGMA)

Fixed: N = 15 teeth, phi = 20 deg, steel-on-steel, T = 50 N*m
Design variables: module m (mm), face width b (mm)

Validation (Shigley's 10th Ed, Ex 14-3 analogue):
  m=3, b=45, T=50 Nm -> sigma_b ~ 119 MPa, sigma_c ~ 506 MPa
"""

import numpy as np

# ── Fixed problem parameters ───────────────────────────────────────────────────
N_TEETH     = 15
PHI         = np.radians(20.0)         # standard pressure angle
GEAR_RATIO  = 3.0                      # mating gear: N_gear = 45 teeth
LEWIS_Y     = 0.290                    # Lewis form factor, N=15, phi=20 deg (Shigley Table 14-2)
C_P         = 191.0                    # elastic coefficient, sqrt(MPa), steel-steel (AGMA 2101)
K_O         = 1.25                     # overload factor (moderate shock, AGMA)
RHO         = 7850.0                   # steel density, kg/m^3

# AGMA geometry factor I for pitting resistance (external spur gear)
# I = (cos(phi)*sin(phi)/2) * (m_G / (m_G+1))
I_FACTOR    = (np.cos(PHI) * np.sin(PHI) / 2.0) * (GEAR_RATIO / (GEAR_RATIO + 1.0))

# Material: AISI 4140 through-hardened (250 HB)
# Allowable bending:  sigma_F0 ~ 0.533*HB + 88.3 = 221.6 MPa, SF=1.2 -> 180 MPa
# Allowable contact:  sigma_H0 ~ 2.22*HB + 200  = 755 MPa,  SH=1.2 -> 524, round to 550
SIGMA_B_ALLOW = 180.0   # MPa
SIGMA_C_ALLOW = 550.0   # MPa

# Standard metric modules (ISO 54)
STD_MODULES = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0]


def simulate(params: dict) -> dict:
    """
    Returns gear stress and mass for given module and face width.

    params keys:
        m      -- module (mm)
        b      -- face width (mm)
        torque -- input torque (N*m, default 50)

    returns dict:
        bending_stress   -- MPa
        contact_stress   -- MPa
        mass             -- kg  (solid cylinder at pitch diameter)
        pitch_diam_mm    -- mm
        tangential_force -- N
        face_ratio       -- b/m  (guideline: 8-16)
    """
    m      = params['m']
    b      = params['b']
    T      = params.get('torque', 50.0)     # N*m

    d   = m * N_TEETH                       # pitch diameter, mm
    F_t = 2000.0 * T / d                    # tangential force, N (T in N*mm = T*1000)

    # Bending stress (Lewis equation)
    sigma_b = F_t / (b * m * LEWIS_Y)      # N/mm^2 = MPa

    # Contact (Hertz) stress, units: C_p in sqrt(MPa), F_t/b/d in MPa -> result MPa
    sigma_c = C_P * np.sqrt(F_t * K_O / (b * d * I_FACTOR))

    # Mass: solid cylinder at pitch diameter
    mass = RHO * np.pi / 4.0 * (d * 1e-3)**2 * (b * 1e-3)   # kg

    return {
        'bending_stress':   float(sigma_b),
        'contact_stress':   float(sigma_c),
        'mass':             float(mass),
        'pitch_diam_mm':    float(d),
        'tangential_force': float(F_t),
        'face_ratio':       float(b / m),
    }


def nearest_std_module(m: float) -> float:
    """Round continuous optimal module to nearest standard ISO value."""
    return min(STD_MODULES, key=lambda s: abs(s - m))


def _feasible(result: dict) -> bool:
    return (result['bending_stress'] <= SIGMA_B_ALLOW and
            result['contact_stress'] <= SIGMA_C_ALLOW)


# ── Quick sanity check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    r = simulate({'m': 3.0, 'b': 45.0})
    print(f"m=3, b=45:  sigma_b={r['bending_stress']:.1f} MPa  "
          f"sigma_c={r['contact_stress']:.1f} MPa  "
          f"mass={r['mass']*1000:.1f} g  "
          f"feasible={'YES' if _feasible(r) else 'NO'}")
    r2 = simulate({'m': 3.5, 'b': 45.0})
    print(f"m=3.5,b=45: sigma_b={r2['bending_stress']:.1f} MPa  "
          f"sigma_c={r2['contact_stress']:.1f} MPa  "
          f"mass={r2['mass']*1000:.1f} g  "
          f"feasible={'YES' if _feasible(r2) else 'NO'}")
