"""Reusable radar range-Doppler CFAR detector (Phase 0 primitive, packaged).

detect(radar_idx) -> structured detections with range, azimuth, aliased radial
velocity, and power. Detection happens in the range-Doppler map (each target a
point with a velocity), NOT the range-azimuth BEV (which smears into arcs).
"""
import numpy as np
from common import H5, radar_config   # sets up sys.path (ROOT + sync/src) + chdir
from utils.cfar import cfar
from lib.bev_render import (load_radar_adc_frame_at_packet_row, sync_radar_idx_to_packet_row,
                            _adc_to_range_doppler, resolve_radar_processing)

SYNC_PPF = 45
DET_DT = {"names": ["range", "az", "vel", "power"], "formats": ["f8"] * 4}


class RadarDetector:
    def __init__(self, processing="ti", max_range_m=25.0,
                 cfar_guard=2, cfar_train=8, cfar_pfa=1e-2, floor_pct=99.0, dealias_margin=1.5):
        self.r = radar_config()
        self.dealias_margin = dealias_margin           # accept de-aliased k only if it beats k=0 by this factor
        self.rres = self.r.range_resolution
        # correct Doppler resolution (config's is the known-buggy one)
        self.lmbda = 3e8 / (self.r.start_freq * 1e9 + self.r.bandwidth / 2)
        self.T_loop = self.r.num_tx * self.r.chirp_interval
        self.sv = self.r.steering_vectors[1]
        self.angs = np.linspace(-90, 90, self.sv.shape[0])
        self.proc = resolve_radar_processing(processing)
        self.maxR = int(max_range_m / self.rres)
        self.g, self.t, self.pfa, self.floor_pct = cfar_guard, cfar_train, cfar_pfa, floor_pct

    def range_doppler(self, radar_idx):
        prow = sync_radar_idx_to_packet_row(int(radar_idx), SYNC_PPF)
        adc, _ = load_radar_adc_frame_at_packet_row(H5, prow, self.r)
        return _adc_to_range_doppler(adc.astype(np.float64), self.proc)  # (nR, nA, nD)

    def _dealias(self, snap, di, nD, dres):
        """TDM-MIMO velocity de-aliasing for one detection snapshot (nA virtual antennas).

        Single-frame Doppler aliases at +-nD/2*dres (=+-2.44 m/s here) because TDM samples each
        virtual channel only every num_tx*chirp_interval. The true velocity is aliased + k*nD*dres
        for k in {-(ntx//2)..ntx//2}. For each k, apply the inter-TX Doppler-phase compensation
        (TX g fired g*chirp_interval late -> extra phase 2*pi*dshift*g/(nD*ntx)) and beamform;
        the correct k makes the virtual array COHERENT -> sharpest beam. Extends +-2.44 -> +-7.32 m/s.
        Returns (az_deg, true_vel). Also fixes the (previously uncompensated) moving-target angle.
        """
        ntx, nrx = self.r.num_tx, self.r.num_rx
        g = np.arange(snap.shape[0]) // nrx                  # TX-group index per virtual antenna
        res = {}
        for k in range(-(ntx // 2), ntx // 2 + 1):
            dshift = (di - nD // 2) + k * nD
            s = snap * np.exp(-1j * 2 * np.pi * dshift * g / (nD * ntx))
            beam = np.abs(self.sv @ s) ** 2
            res[k] = (float(beam.max()), self.angs[int(np.argmax(beam))], dshift * dres)
        p0, az0, v0 = res[0]                                 # the no-de-alias (k=0) hypothesis
        # CONSERVATIVE: only accept a shifted interval if it beats k=0 by a clear coherence margin
        # (12-element array -> k hypotheses are close; without this, noise de-aliases STATIC returns).
        bk = max((k for k in res if k != 0), key=lambda k: res[k][0], default=None)
        if bk is not None and res[bk][0] > p0 * self.dealias_margin:
            return res[bk][1], res[bk][2]
        return az0, v0

    def detect(self, radar_idx, min_abs_vel=0.0, dealias=False):   # opt-in: helps fast clean movers,
        # but the 12-element array makes the per-detection k-pick noisy on mostly-static scenes
        rd = self.range_doppler(radar_idx)
        nR, nA, nD = rd.shape
        dres = self.lmbda / (2 * nD * self.T_loop)
        rdp_db = 10 * np.log10(np.sum(np.abs(rd) ** 2, axis=1) + 1e-9)
        det, _, _ = cfar(self.g, self.t, self.pfa, rdp_db[:self.maxR])
        ri, di = np.where(det > 0)
        if not len(ri):
            return np.zeros(0, dtype=DET_DT), dres
        rng = ri * self.rres
        pw = rdp_db[ri, di]
        if dealias:                                          # de-aliased velocity + coherent azimuth
            az = np.empty(len(ri)); vel = np.empty(len(ri))
            for n, (r, d) in enumerate(zip(ri, di)):
                az[n], vel[n] = self._dealias(rd[r, :, d], d, nD, dres)
        else:                                                # legacy: aliased velocity, uncompensated angle
            vel = (di - nD // 2) * dres
            az = np.array([self.angs[np.argmax(np.abs(self.sv @ rd[r, :, d]) ** 2)]
                           for r, d in zip(ri, di)])
        keep = (rng >= 2) & (rng <= self.maxR * self.rres) & \
               (pw > np.percentile(rdp_db, self.floor_pct)) & (np.abs(vel) >= min_abs_vel)
        out = np.zeros(keep.sum(), dtype=DET_DT)
        out["range"], out["az"], out["vel"], out["power"] = rng[keep], az[keep], vel[keep], pw[keep]
        return out, dres

    @staticmethod
    def to_xy(dets):
        """radar-frame BEV: forward X, lateral Y (azimuth sign-corrected for BEV fliplr)."""
        azr = np.radians(-dets["az"])
        return dets["range"] * np.cos(azr), dets["range"] * np.sin(azr)
