"""Write schematic waveform and probability curves to ``curvas.tex``."""

import numpy as np

def coords(xs, ys, prec=3):
    """Format paired arrays as TikZ coordinate pairs."""
    return " ".join(f"({x:.{prec}f},{y:.{prec}f})" for x, y in zip(xs, ys))

def traza(x0, x1, xP, xS, n=420, amp=0.42, seed=7):
    """Generate a schematic trace with a P arrival and optional S arrival."""
    rng = np.random.default_rng(seed)
    x = np.linspace(x0, x1, n)
    y = 0.035 * np.sin(2*np.pi*3.1*x + 0.7) + 0.025 * rng.standard_normal(n)
    y = np.convolve(y, np.ones(5)/5, mode="same")
    mP = x >= xP
    tP = x[mP] - xP
    y[mP] += amp*(1-np.exp(-tP*9))*np.exp(-tP*0.85)*np.sin(2*np.pi*4.4*tP)
    if xS is not None:
        mS = x >= xS
        tS = x[mS] - xS
        y[mS] += 1.35*amp*(1-np.exp(-tS*7))*np.exp(-tS*0.6)*np.sin(2*np.pi*2.6*tS+0.4)
    return x, y

def gauss(xc, sig, h, x0, x1, n=200):
    """Generate a Gaussian curve over the requested horizontal interval."""
    x = np.linspace(x0, x1, n)
    return x, h*np.exp(-0.5*((x-xc)/sig)**2)

out = {}
x, y = traza(0.0, 6.4, 2.6, None, amp=0.40)
out["TRAZA"] = coords(x, y)

for nombre, sig in (("GAUSS", 0.42), ("GAUSSNARROW", 0.16), ("GAUSSWIDE", 1.05)):
    gx, gy = gauss(2.6, sig, 1.05, 0.0, 6.4)
    out[nombre] = coords(gx, gy)

# Output-panel width and peak positions.
W = 1.95
def pico(xc, sig=0.13, h=0.5, n=140):
    """Format a narrow probability peak for an output panel."""
    x = np.linspace(0, W, n)
    return coords(x, h*np.exp(-0.5*((x-xc)/sig)**2))

XP, XS = 0.75*W/2.7*2.7*0.28, 0.62*W   # P near 28%, S near 62%
XP = 0.28*W
out["PICOP"] = pico(XP)
out["PICOS"] = pico(XS)

# Schematic noise probability falls near the two phase peaks.
xx = np.linspace(0, W, 160)
yy = 0.5*(1 - np.exp(-0.5*((xx-XP)/0.13)**2) - np.exp(-0.5*((xx-XS)/0.13)**2))
out["PICON"] = coords(xx, np.clip(yy, 0, None))

out["XP"] = f"{XP:.3f}"
out["XS"] = f"{XS:.3f}"
out["W"]  = f"{W:.3f}"

with open("curvas.tex", "w") as f:
    for k, v in out.items():
        f.write(f"\\def\\{k}{{{v}}}\n")
print("ok", {k: (len(v) if len(v) > 20 else v) for k, v in out.items()})
