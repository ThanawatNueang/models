"""Small vectorized 3D noise library (value noise, fbm, ridged, voronoi-ish).

All functions take an (N,3) float array of positions (metres) and return (N,).
"""
import numpy as np

_rng = np.random.default_rng(1234)
_PERM = _rng.permutation(256)
_PERM = np.concatenate([_PERM, _PERM]).astype(np.int64)
_VALS = _rng.random(256)


def _hash(ix, iy, iz):
    return _VALS[_PERM[(_PERM[(_PERM[ix & 255] + iy) & 255] + iz) & 255]]


def value(p, scale=1.0, seed=0):
    """Smooth value noise in [0,1]. `scale` = features per metre."""
    q = p * scale + seed * 17.123
    i = np.floor(q).astype(np.int64)
    f = q - i
    u = f * f * f * (f * (f * 6 - 15) + 10)
    x0, y0, z0 = i[:, 0], i[:, 1], i[:, 2]
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]

    def lerp(a, b, t):
        return a + (b - a) * t

    c00 = lerp(_hash(x0, y0, z0), _hash(x1, y0, z0), ux)
    c10 = lerp(_hash(x0, y1, z0), _hash(x1, y1, z0), ux)
    c01 = lerp(_hash(x0, y0, z1), _hash(x1, y0, z1), ux)
    c11 = lerp(_hash(x0, y1, z1), _hash(x1, y1, z1), ux)
    return lerp(lerp(c00, c10, uy), lerp(c01, c11, uy), uz)


def fbm(p, scale=1.0, octaves=4, lac=2.0, gain=0.5, seed=0):
    """Fractal noise in roughly [0,1]."""
    total, amp, norm = 0.0, 1.0, 0.0
    s = scale
    for o in range(octaves):
        total = total + value(p, s, seed + o * 31) * amp
        norm += amp
        amp *= gain
        s *= lac
    return total / norm


def ridged(p, scale=1.0, octaves=3, seed=0):
    """Thin ridge lines (veins, cracks) in [0,1], 1 on the ridge."""
    total, amp, norm = 0.0, 1.0, 0.0
    s = scale
    for o in range(octaves):
        n = 1 - np.abs(value(p, s, seed + o * 13) * 2 - 1)
        total = total + n ** 6 * amp
        norm += amp
        amp *= 0.5
        s *= 2.1
    return total / norm


def stretched(p, axis, scale_along, scale_across, octaves=3, seed=0):
    """Noise stretched along `axis` (unit (3,) or (N,3)): hair strands, brushed fabric."""
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim == 1:
        axis = np.broadcast_to(axis, p.shape)
    along = (p * axis).sum(axis=1, keepdims=True)
    across = p - along * axis
    q = across * scale_across + along * axis * scale_along
    return fbm(q, 1.0, octaves, seed=seed)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)
