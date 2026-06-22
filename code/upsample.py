import numpy as np

def upsample_rdr_sparse_xyzpw(
    pts_xyzw,                      # (N,4) float32: [x, y, z, pw]
    base_samples_per_point=8,      # baseline samples per original point
    sigma_xyz=(0.30, 0.30, 0.50),  # Gaussian std (m) along x/y/z
    bev_cell=(0.5, 0.5),           # (dy, dx) for BEV density modulation in meters
    bev_extent=((-60., 60.), (0., 120.)),  # ((y_min,y_max),(x_min,x_max)) meters
    density_exponent=1.0,          # 0→ignore BEV; >1 favors dense cells; <1 flattens
    include_original=True,         # keep the original points
    power_mode="multiply",         # "multiply" (pw * kernel) or "copy" (unchanged pw)
    kernel="gaussian",             # "gaussian" or "sinc2"
    sinc_w=1.0,                    # main-lobe width for sinc2 kernel, in meters (if used)
    power_noise_std=0.0,           # optional additive noise on power
    min_power=None, max_power=None,# optional power clipping
    rng=None
):
    """
    Returns:
        (M,4) float32 array of [x, y, z, pw] after upsampling.

    Notes:
    - Gaussian kernel: k = exp(-0.5 * (dx^2/sx^2 + dy^2/sy^2 + dz^2/sz^2))
    - Sinc^2 kernel  : k = sinc((dx)/w)^2 * sinc((dy)/w)^2   [z left Gaussian]
      Use this when you want a physically-inspired lateral spread; tune 'sinc_w'.
    """
    if rng is None:
        rng = np.random.default_rng()

    pts = np.asarray(pts_xyzw, dtype=np.float32)
    assert pts.ndim == 2 and pts.shape[1] == 4, "pts must be (N,4) [x,y,z,pw]"
    N = pts.shape[0]
    if N == 0:
        return pts

    sx, sy, sz = map(float, sigma_xyz)
    y_min, y_max = bev_extent[0]
    x_min, x_max = bev_extent[1]
    cy, cx = bev_cell
    nx = int(np.ceil((x_max - x_min) / cx))
    ny = int(np.ceil((y_max - y_min) / cy))

    x = pts[:, 0]; y = pts[:, 1]
    ix = ((x - x_min) / cx).astype(int)
    iy = ((y - y_min) / cy).astype(int)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    # --- BEV density map (for sample count modulation) ---
    if np.any(valid):
        counts = np.zeros((ny, nx), dtype=np.int32)
        for j in np.where(valid)[0]:
            counts[iy[j], ix[j]] += 1
        c = np.zeros(N, dtype=np.float32)
        v = np.where(valid)[0]
        c[v] = counts[iy[v], ix[v]].astype(np.float32)
        if c.max() > 0:
            c = c / c.max()
        weights = np.power(np.clip(c, 1e-6, 1.0), density_exponent).astype(np.float32)
    else:
        weights = np.ones(N, dtype=np.float32)

    samples_per_point = np.maximum(0, np.round(base_samples_per_point * weights)).astype(int)
    if samples_per_point.sum() == 0:
        samples_per_point = np.ones(N, dtype=int)

    total_new = int(samples_per_point.sum())
    out_sz = total_new + (N if include_original else 0)
    out = np.empty((out_sz, 4), dtype=np.float32)

    # precalc inverses for Gaussian
    inv2_sx2 = 1.0 / (2.0 * sx * sx) if sx > 0 else np.inf
    inv2_sy2 = 1.0 / (2.0 * sy * sy) if sy > 0 else np.inf
    inv2_sz2 = 1.0 / (2.0 * sz * sz) if sz > 0 else np.inf

    write = 0
    use_gauss = (kernel.lower() == "gaussian")
    use_sinc2 = (kernel.lower() == "sinc2")
    sinc_w = float(sinc_w) if sinc_w is not None else 1.0

    for i in range(N):
        k = samples_per_point[i]
        if k <= 0:
            continue
        xi, yi, zi, pwi = pts[i]

        dx = rng.normal(0.0, sx, size=k).astype(np.float32)
        dy = rng.normal(0.0, sy, size=k).astype(np.float32)
        dz = rng.normal(0.0, sz, size=k).astype(np.float32)

        xs = xi + dx
        ys = yi + dy
        zs = zi + dz

        if use_gauss:
            kval = np.exp(-(dx*dx)*inv2_sx2 - (dy*dy)*inv2_sy2 - (dz*dz)*inv2_sz2).astype(np.float32)
        elif use_sinc2:
            # numpy.sinc(u) = sin(pi*u)/(pi*u)
            # Apply sinc^2 laterally; keep Gaussian along z to avoid infinite support
            kx = np.sinc(dx / max(sinc_w, 1e-6))
            ky = np.sinc(dy / max(sinc_w, 1e-6))
            kz = np.exp(-(dz*dz)*inv2_sz2)  # gentle confinement in z
            kval = (kx*kx * ky*ky * kz).astype(np.float32)
        else:
            raise ValueError("kernel must be 'gaussian' or 'sinc2'")

        if power_mode == "multiply":
            ps = (pwi * kval).astype(np.float32)
        elif power_mode == "copy":
            ps = np.full(k, pwi, dtype=np.float32)
        else:
            raise ValueError("power_mode must be 'multiply' or 'copy'")

        if power_noise_std and power_noise_std > 0:
            ps += rng.normal(0.0, power_noise_std, size=k).astype(np.float32)

        if (min_power is not None) or (max_power is not None):
            lo = -np.inf if min_power is None else min_power
            hi =  np.inf if max_power is None else max_power
            ps = np.clip(ps, lo, hi, out=ps)

        out[write:write+k, 0] = xs
        out[write:write+k, 1] = ys
        out[write:write+k, 2] = zs
        out[write:write+k, 3] = ps
        write += k

    if include_original:
        out[write:write+N] = pts
        write += N

    return out[:write]