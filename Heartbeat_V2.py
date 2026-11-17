import sys
import math
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


# =============================================================================
# MATHEMATICAL & GEOMETRIC UTILITIES
# =============================================================================

def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    return v / norm


def smoothstep01(x):
    """C1-continuous cubic Hermite interpolation on [0, 1]."""
    x_c = np.clip(x, 0.0, 1.0)
    return x_c * x_c * (3.0 - 2.0 * x_c)


def cosine_window(phase, start, peak_start, peak_end, end):
    """Smooth C1 plateau window function for physiological valve opening."""
    if phase <= start or phase >= end:
        return 0.0
    if phase < peak_start:
        t = (phase - start) / (peak_start - start)
        return 0.5 * (1.0 - math.cos(t * math.pi))
    if phase <= peak_end:
        return 1.0
    t = (phase - peak_end) / (end - peak_end)
    return 0.5 * (1.0 + math.cos(t * math.pi))


def catmull_rom_spline(points, num_samples=60):
    """Evaluates a smooth 3D Catmull-Rom spline through control points."""
    pts = np.asarray(points, dtype=np.float32)
    n = len(pts)
    if n < 2:
        return pts

    p_ext = np.vstack([pts[0] - (pts[1] - pts[0]), pts, pts[-1] + (pts[-1] - pts[-2])])
    total_segments = n - 1
    t_global = np.linspace(0, total_segments, num_samples, endpoint=True)

    out = np.zeros((num_samples, 3), dtype=np.float32)
    for i, tg in enumerate(t_global):
        seg = min(int(np.floor(tg)), total_segments - 1)
        t = tg - seg
        p0, p1, p2, p3 = p_ext[seg], p_ext[seg + 1], p_ext[seg + 2], p_ext[seg + 3]
        t2 = t * t
        t3 = t2 * t
        out[i] = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
    return out


def compute_vertex_normals(vertices, indices):
    """Computes smooth area-weighted per-vertex normals for a triangle mesh."""
    normals = np.zeros_like(vertices, dtype=np.float32)
    tris = indices.reshape(-1, 3)
    v0 = vertices[tris[:, 0]]
    v1 = vertices[tris[:, 1]]
    v2 = vertices[tris[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    np.add.at(normals, tris[:, 0], face_normals)
    np.add.at(normals, tris[:, 1], face_normals)
    np.add.at(normals, tris[:, 2], face_normals)
    return normalize(normals)


def build_tube_mesh(centerline, radii, radial_segments=20, arc_range=(0.0, 2.0 * math.pi)):
    """
    Constructs a smooth tubular or partial-shell 3D mesh along a 3D centerline
    using parallel transport frames to prevent twisting singularities.
    """
    pts = np.asarray(centerline, dtype=np.float32)
    n_rings = len(pts)
    if np.isscalar(radii):
        radii_arr = np.full(n_rings, radii, dtype=np.float32)
    else:
        radii_arr = np.asarray(radii, dtype=np.float32)

    tangents = np.zeros_like(pts)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    tangents = normalize(tangents)

    normals = np.zeros_like(pts)
    binormals = np.zeros_like(pts)
    init_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(np.dot(tangents[0], init_up)) > 0.9:
        init_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    normals[0] = normalize(np.cross(tangents[0], init_up))
    binormals[0] = normalize(np.cross(tangents[0], normals[0]))

    for i in range(1, n_rings):
        axis = np.cross(tangents[i - 1], tangents[i])
        sin_a = np.linalg.norm(axis)
        cos_a = np.clip(np.dot(tangents[i - 1], tangents[i]), -1.0, 1.0)
        if sin_a > 1e-6:
            axis = axis / sin_a
            angle = math.atan2(sin_a, cos_a)
            n_prev = normals[i - 1]
            normals[i] = (
                n_prev * math.cos(angle)
                + np.cross(axis, n_prev) * math.sin(angle)
                + axis * np.dot(axis, n_prev) * (1.0 - math.cos(angle))
            )
            normals[i] = normalize(normals[i])
        else:
            normals[i] = normals[i - 1]
        binormals[i] = normalize(np.cross(tangents[i], normals[i]))

    full_circle = abs((arc_range[1] - arc_range[0]) - 2.0 * math.pi) < 1e-4
    angles = np.linspace(arc_range[0], arc_range[1], radial_segments, endpoint=not full_circle)
    n_cols = len(angles)

    vertices = np.zeros((n_rings * n_cols, 3), dtype=np.float32)
    for i in range(n_rings):
        for j, theta in enumerate(angles):
            r = radii_arr[i]
            radial_vec = math.cos(theta) * normals[i] + math.sin(theta) * binormals[i]
            vertices[i * n_cols + j] = pts[i] + r * radial_vec

    indices = []
    for i in range(n_rings - 1):
        for j in range(n_cols if full_circle else n_cols - 1):
            jn = (j + 1) % n_cols
            a = i * n_cols + j
            b = (i + 1) * n_cols + j
            c = (i + 1) * n_cols + jn
            d = i * n_cols + jn
            indices.extend([a, b, c, a, c, d])

    indices = np.array(indices, dtype=np.uint32)
    v_normals = compute_vertex_normals(vertices, indices)
    return vertices, v_normals, indices


# =============================================================================
# PROCEDURAL CARDIAC AUSCULTATION AUDIO ENGINE (LUB-DUB + BLOOD RUSH)
# =============================================================================

class CardiacAudioEngine:
    """
    Synthesizes realistic physiological heart sounds in memory using NumPy & Pygame:
      - S1 ("Lub"): Low-pitched dual-component M1/T1 thud when Mitral & Tricuspid valves close.
      - S2 ("Dub"): Higher-pitched crisp A2/P2 snap when Aortic & Pulmonary valves close.
      - Systolic Flow Whoosh: Soft band-limited arterial blood rush during ventricular ejection.
    """
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.enabled = True
        self.available = False
        self.active_sound_label = "SILENT"
        self.label_timer = 0.0

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=512)
            self.available = True
            self.s1_sound = self._synthesize_s1_lub()
            self.s2_sound = self._synthesize_s2_dub()
            self.flow_sound = self._synthesize_systolic_whoosh()
        except Exception as e:
            print(f"[Audio Warning] Could not initialize sound mixer: {e}")
            self.available = False

    def _to_stereo_sound(self, mono_wave, volume=0.85):
        """Normalizes a 1D float array and converts it into a 16-bit stereo Pygame Sound."""
        peak = np.max(np.abs(mono_wave))
        if peak > 1e-6:
            mono_wave = (mono_wave / peak) * volume
        pcm16 = np.clip(mono_wave * 32767.0, -32767, 32767).astype(np.int16)
        stereo = np.column_stack((pcm16, pcm16))
        return pygame.sndarray.make_sound(stereo)

    def _synthesize_s1_lub(self):
        """
        Synthesizes S1 ('Lub'): Mitral (M1) + Tricuspid (T1) valve closure + myocardial tension.
        Duration: ~0.13s, warm low-frequency stethoscope chest-piece resonance (38-75 Hz).
        """
        duration = 0.135
        t = np.linspace(0.0, duration, int(self.sample_rate * duration), endpoint=False)

        # M1 (Mitral closure at t=0.0s)
        env1 = (t / 0.012) * np.exp(-t / 0.022)
        phase1 = 2.0 * np.pi * (68.0 * t - 110.0 * (t ** 2))
        m1 = env1 * (np.sin(phase1) + 0.45 * np.sin(1.85 * phase1))

        # T1 (Tricuspid closure delayed by ~22ms)
        t2 = np.maximum(0.0, t - 0.022)
        env2 = (t2 / 0.012) * np.exp(-t2 / 0.024)
        phase2 = 2.0 * np.pi * (58.0 * t2 - 90.0 * (t2 ** 2))
        t1 = 0.78 * env2 * (np.sin(phase2) + 0.35 * np.sin(1.9 * phase2))

        # Subtle low-frequency valve tissue thump
        rng = np.random.default_rng(101)
        noise = rng.normal(0.0, 1.0, len(t))
        kernel = np.hanning(45)
        kernel /= np.sum(kernel)
        thump = np.convolve(noise, kernel, mode="same") * env1 * 0.35

        wave = m1 + t1 + thump
        # Smooth fade-out at tail to prevent any click
        wave *= np.sin(np.linspace(0.0, np.pi, len(t))) ** 0.25
        return self._to_stereo_sound(wave, volume=0.92)

    def _synthesize_s2_dub(self):
        """
        Synthesizes S2 ('Dub'): Aortic (A2) + Pulmonary (P2) semilunar valve snap.
        Duration: ~0.10s, firmer, slightly higher frequency (55-110 Hz).
        """
        duration = 0.105
        t = np.linspace(0.0, duration, int(self.sample_rate * duration), endpoint=False)

        # A2 (Aortic valve snap at t=0.0s)
        env1 = (t / 0.008) * np.exp(-t / 0.016)
        phase1 = 2.0 * np.pi * (96.0 * t - 180.0 * (t ** 2))
        a2 = env1 * (np.sin(phase1) + 0.50 * np.sin(1.75 * phase1))

        # P2 (Pulmonary valve snap delayed by ~24ms)
        t2 = np.maximum(0.0, t - 0.024)
        env2 = (t2 / 0.008) * np.exp(-t2 / 0.017)
        phase2 = 2.0 * np.pi * (82.0 * t2 - 150.0 * (t2 ** 2))
        p2 = 0.72 * env2 * (np.sin(phase2) + 0.40 * np.sin(1.8 * phase2))

        rng = np.random.default_rng(202)
        noise = rng.normal(0.0, 1.0, len(t))
        kernel = np.hanning(31)
        kernel /= np.sum(kernel)
        snap = np.convolve(noise, kernel, mode="same") * env1 * 0.28

        wave = a2 + p2 + snap
        wave *= np.sin(np.linspace(0.0, np.pi, len(t))) ** 0.25
        return self._to_stereo_sound(wave, volume=0.82)

    def _synthesize_systolic_whoosh(self):
        """
        Synthesizes a soft band-limited Doppler/stethoscope blood ejection rush
        heard as blood surges into the Aorta and Pulmonary Trunk.
        """
        duration = 0.19
        t = np.linspace(0.0, duration, int(self.sample_rate * duration), endpoint=False)
        rng = np.random.default_rng(303)
        white = rng.normal(0.0, 1.0, len(t))

        # Low-pass filter via moving window convolution for a warm fluid rush
        kernel = np.hanning(65)
        kernel /= np.sum(kernel)
        filtered = np.convolve(white, kernel, mode="same")

        env = np.sin((t / duration) * np.pi) ** 1.6
        pulse_mod = 0.65 + 0.35 * np.sin(2.0 * np.pi * 38.0 * t)
        wave = filtered * env * pulse_mod
        return self._to_stereo_sound(wave, volume=0.22)

    def update_and_trigger(self, prev_phase, curr_phase, dt, paused):
        if self.label_timer > 0.0:
            self.label_timer = max(0.0, self.label_timer - dt)
            if self.label_timer == 0.0:
                self.active_sound_label = "SILENT"

        if paused or not self.enabled or not self.available or dt <= 0.0:
            return

        # Helper to detect phase crossing across wrap-around [0..1]
        def crossed(target):
            if prev_phase <= curr_phase:
                return prev_phase < target <= curr_phase
            return prev_phase < target or curr_phase >= target

        # 1. Trigger S1 ("LUB") when Tricuspid & Mitral valves snap shut (phase = 0.14)
        if crossed(0.14):
            self.s1_sound.play()
            self.active_sound_label = "S1: 'LUB' (AV Valves Close)"
            self.label_timer = 0.18

        # 2. Trigger subtle Systolic Ejection Rush during peak outflow (phase = 0.22)
        if crossed(0.22):
            self.flow_sound.play()

        # 3. Trigger S2 ("DUB") when Aortic & Pulmonary valves snap shut (phase = 0.46)
        if crossed(0.46):
            self.s2_sound.play()
            self.active_sound_label = "S2: 'DUB' (Semilunar Close)"
            self.label_timer = 0.16


# =============================================================================
# PHYSIOLOGICAL CARDIAC CYCLE ENGINE
# =============================================================================

class CardiacCycleEngine:
    """
    Computes smooth C1-continuous physiological parameters of the human cardiac cycle:
      - Atrial Systole (0.00 - 0.14): Atria contract, AV valves wide open.
      - Isovolumetric Contraction (0.14 - 0.19): AV valves close (S1 'Lub'), ventricles tense.
      - Ventricular Ejection / Systole (0.19 - 0.48): Semilunar valves open, ventricles contract/twist.
      - Isovolumetric Relaxation (0.48 - 0.54): Semilunar valves close (S2 'Dub', dicrotic notch).
      - Passive Ventricular Filling / Diastole (0.54 - 1.00): AV valves open, chambers refill.
    """
    def __init__(self, bpm=72.0):
        self.bpm = bpm
        self.period = 60.0 / bpm
        self.time = 0.0
        self.prev_phase = 0.0
        self.phase = 0.0

        self.atrial_contraction = 0.0
        self.ventricular_contraction = 0.0
        self.arterial_pulse = 0.0
        self.apical_twist = 0.0

        self.tricuspid_open = 1.0
        self.mitral_open = 1.0
        self.pulmonary_open = 0.0
        self.aortic_open = 0.0

        self.phase_name = "Diastole (Passive Filling)"
        self.ecg_history = [0.0] * 180
        self.pcg_history = [0.0] * 180
        self.ecg_accum = 0.0

    def reset(self):
        self.time = 0.0
        self.prev_phase = 0.0
        self.phase = 0.0
        self.ecg_accum = 0.0
        self.ecg_history = [0.0] * 180
        self.pcg_history = [0.0] * 180
        self.update(0.0)

    def _compute_ecg(self, p):
        """Synthesizes a clinical P-QRS-T electrocardiogram waveform."""
        p_wave = 0.18 * math.exp(-((p - 0.06) / 0.032) ** 2)
        q_wave = -0.15 * math.exp(-((p - 0.155) / 0.010) ** 2)
        r_wave = 1.05 * math.exp(-((p - 0.175) / 0.012) ** 2)
        s_wave = -0.28 * math.exp(-((p - 0.195) / 0.011) ** 2)
        t_wave = 0.30 * math.exp(-((p - 0.41) / 0.048) ** 2)
        return p_wave + q_wave + r_wave + s_wave + t_wave

    def _compute_pcg(self, p):
        """Synthesizes a visual Phonocardiogram (PCG) waveform showing S1 (Lub) and S2 (Dub) bursts."""
        s1_env = math.exp(-((p - 0.16) / 0.026) ** 2)
        s2_env = 0.82 * math.exp(-((p - 0.475) / 0.020) ** 2)
        s1_osc = s1_env * math.sin(2.0 * math.pi * 42.0 * p)
        s2_osc = s2_env * math.sin(2.0 * math.pi * 64.0 * p)
        return s1_osc + s2_osc

    def update(self, dt):
        self.prev_phase = self.phase
        self.time += dt
        self.phase = (self.time % self.period) / self.period
        p = self.phase

        # 1. Smooth Atrial Contraction Profile
        if p < 0.15:
            self.atrial_contraction = math.sin((p / 0.15) * math.pi) ** 1.4
        else:
            self.atrial_contraction = -0.07 * math.sin(((p - 0.15) / 0.85) * math.pi)

        # 2. Smooth Ventricular Contraction Profile
        if 0.14 <= p < 0.52:
            norm_v = (p - 0.14) / 0.38
            self.ventricular_contraction = math.sin(norm_v * math.pi) ** 1.25
        else:
            if p < 0.14:
                self.ventricular_contraction = -0.06 * math.sin((p / 0.14) * math.pi)
            else:
                self.ventricular_contraction = 0.0

        # 3. Smooth Arterial Pulse Wave (Aorta & Pulmonary Trunk expansion)
        if 0.19 <= p < 0.64:
            norm_a = (p - 0.19) / 0.45
            main_wave = math.sin(norm_a * math.pi) ** 1.4
            dicrotic = 0.16 * math.exp(-((p - 0.52) / 0.038) ** 2)
            self.arterial_pulse = min(1.0, main_wave + dicrotic)
        else:
            self.arterial_pulse = 0.0

        self.apical_twist = 0.22 * self.ventricular_contraction

        # 4. C1-Smooth Atrioventricular Valves (Tricuspid & Mitral)
        if p < 0.13:
            av_open = 0.82 + 0.18 * math.sin((p / 0.13) * math.pi)
        elif p < 0.175:
            t = (p - 0.13) / 0.045
            av_open = 0.82 * 0.5 * (1.0 + math.cos(t * math.pi))
        elif p < 0.535:
            av_open = 0.0
        elif p < 0.595:
            t = (p - 0.535) / 0.060
            av_open = 0.82 * 0.5 * (1.0 - math.cos(t * math.pi))
        else:
            av_open = 0.82 + 0.10 * math.sin(((p - 0.595) / 0.405) * math.pi)

        self.tricuspid_open = float(np.clip(av_open, 0.0, 1.0))
        self.mitral_open = float(np.clip(av_open, 0.0, 1.0))

        # 5. C1-Smooth Semilunar Valves (Pulmonary & Aortic)
        sl_open = cosine_window(p, 0.185, 0.23, 0.44, 0.49)
        self.pulmonary_open = float(np.clip(sl_open, 0.0, 1.0))
        self.aortic_open = float(np.clip(sl_open, 0.0, 1.0))

        # Phase Name for HUD
        if p < 0.14:
            self.phase_name = "Atrial Systole (Atria Contract, AV Open)"
        elif p < 0.19:
            self.phase_name = "Isovolumetric Contraction (S1 'Lub')"
        elif p < 0.46:
            self.phase_name = "Ventricular Systole (Ejection, SL Open)"
        elif p < 0.54:
            self.phase_name = "Isovolumetric Relaxation (S2 'Dub')"
        else:
            self.phase_name = "Ventricular Diastole (Filling, AV Open)"

        # Sample ECG & PCG waveforms at fixed intervals
        if dt > 0.0:
            self.ecg_accum += dt
            step = 1.0 / 60.0
            while self.ecg_accum >= step:
                self.ecg_accum -= step
                self.ecg_history.append(self._compute_ecg(p))
                self.pcg_history.append(self._compute_pcg(p))
                if len(self.ecg_history) > 180:
                    self.ecg_history.pop(0)
                if len(self.pcg_history) > 180:
                    self.pcg_history.pop(0)


# =============================================================================
# PRECOMPUTED VECTORIZED DEFORMATION ENGINE (ZERO LAG)
# =============================================================================

def deform_points_direct(pts, cycle: CardiacCycleEngine):
    """Direct deformation for a small array of landmark points (valve centers, papillary tips)."""
    v = pts.copy()
    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    v_mask = smoothstep01((0.40 - y) / 1.85)
    vc = cycle.ventricular_contraction

    vent_center_x = 0.12 * (0.3 - y)
    vent_center_z = 0.08
    dx_v = x - vent_center_x
    dz_v = z - vent_center_z

    squeeze = 1.0 - 0.24 * vc * v_mask
    x_new = vent_center_x + dx_v * squeeze
    z_new = vent_center_z + dz_v * squeeze
    y_new = y + 0.22 * vc * (v_mask ** 1.4)

    twist_angle = cycle.apical_twist * (v_mask ** 1.3)
    cos_t = np.cos(twist_angle)
    sin_t = np.sin(twist_angle)
    rx = x_new - vent_center_x
    rz = z_new - vent_center_z
    x_new = vent_center_x + rx * cos_t - rz * sin_t
    z_new = vent_center_z + rx * sin_t + rz * cos_t

    a_mask = np.exp(-((y - 0.78) / 0.42) ** 2)
    ac = cycle.atrial_contraction
    ra_dist = np.exp(-((x + 1.10) ** 2 + (z - 0.05) ** 2) / 0.95)
    la_dist = np.exp(-((x - 0.85) ** 2 + (z + 0.35) ** 2) / 0.95)

    atrial_pull_ra = 0.22 * ac * a_mask * ra_dist
    atrial_pull_la = 0.22 * ac * a_mask * la_dist

    x_new = x_new - atrial_pull_ra * (x_new + 1.10) - atrial_pull_la * (x_new - 0.85)
    y_new = y_new - 0.12 * ac * a_mask * (ra_dist + la_dist) * (y_new - 0.65)
    z_new = z_new - atrial_pull_ra * (z_new - 0.05) - atrial_pull_la * (z_new + 0.35)

    art_mask = np.clip((y - 0.85) / 1.2, 0.0, 1.0)
    ap = cycle.arterial_pulse
    scale_art = 1.0 + 0.06 * ap * art_mask
    x_new = x_new * scale_art
    z_new = z_new * scale_art

    v[:, 0] = x_new
    v[:, 1] = y_new
    v[:, 2] = z_new
    return v


class FastDeformableBatch:
    """
    Combines multiple 3D sub-meshes into a single contiguous Vertex + Normal + Color buffer
    and precomputes all spatial deformation masks once at startup.
    """
    def __init__(self, mesh_list):
        verts_all = []
        norms_all = []
        cols_all = []
        indices_all = []
        v_offset = 0

        for v, n, idx, col in mesh_list:
            n_v = len(v)
            verts_all.append(v)
            norms_all.append(n)
            c_arr = np.zeros((n_v, 4), dtype=np.float32)
            if len(col) == 3:
                c_arr[:, :3] = col
                c_arr[:, 3] = 1.0
            else:
                c_arr[:] = col
            cols_all.append(c_arr)
            indices_all.append(idx + v_offset)
            v_offset += n_v

        self.base_verts = np.vstack(verts_all).astype(np.float32)
        self.base_norms = np.vstack(norms_all).astype(np.float32)
        self.colors = np.vstack(cols_all).astype(np.float32)
        self.indices = np.concatenate(indices_all).astype(np.uint32)

        x = self.base_verts[:, 0]
        y = self.base_verts[:, 1]
        z = self.base_verts[:, 2]

        self.v_mask = smoothstep01((0.40 - y) / 1.85).astype(np.float32)
        self.v_squeeze_factor = (0.24 * self.v_mask).astype(np.float32)
        self.v_lift_factor = (0.22 * (self.v_mask ** 1.4)).astype(np.float32)
        self.v_twist_factor = (self.v_mask ** 1.3).astype(np.float32)

        self.vent_cx = (0.12 * (0.3 - y)).astype(np.float32)
        self.vent_cz = np.float32(0.08)
        self.dx_v = (x - self.vent_cx).astype(np.float32)
        self.dz_v = (z - self.vent_cz).astype(np.float32)

        a_mask = np.exp(-((y - 0.78) / 0.42) ** 2)
        ra_dist = np.exp(-((x + 1.10) ** 2 + (z - 0.05) ** 2) / 0.95)
        la_dist = np.exp(-((x - 0.85) ** 2 + (z + 0.35) ** 2) / 0.95)

        self.ra_weight = (0.22 * a_mask * ra_dist).astype(np.float32)
        self.la_weight = (0.22 * a_mask * la_dist).astype(np.float32)
        self.ay_weight = (0.12 * a_mask * (ra_dist + la_dist)).astype(np.float32)
        self.art_weight = (0.06 * np.clip((y - 0.85) / 1.2, 0.0, 1.0)).astype(np.float32)

        self.cur_verts = np.empty_like(self.base_verts)
        self.cur_norms = np.empty_like(self.base_norms)

    def update_geometry(self, cycle: CardiacCycleEngine):
        vc = np.float32(cycle.ventricular_contraction)
        ac = np.float32(cycle.atrial_contraction)
        ap = np.float32(cycle.arterial_pulse)
        tw = np.float32(cycle.apical_twist)

        squeeze = 1.0 - vc * self.v_squeeze_factor
        rx = self.dx_v * squeeze
        rz = self.dz_v * squeeze
        y_new = self.base_verts[:, 1] + vc * self.v_lift_factor

        twist_ang = tw * self.v_twist_factor
        cos_t = np.cos(twist_ang)
        sin_t = np.sin(twist_ang)

        x_new = self.vent_cx + rx * cos_t - rz * sin_t
        z_new = self.vent_cz + rx * sin_t + rz * cos_t

        pull_ra = ac * self.ra_weight
        pull_la = ac * self.la_weight

        x_new = x_new - pull_ra * (x_new + 1.10) - pull_la * (x_new - 0.85)
        y_new = y_new - ac * self.ay_weight * (y_new - 0.65)
        z_new = z_new - pull_ra * (z_new - 0.05) - pull_la * (z_new + 0.35)

        art_scale = 1.0 + ap * self.art_weight
        self.cur_verts[:, 0] = x_new * art_scale
        self.cur_verts[:, 1] = y_new
        self.cur_verts[:, 2] = z_new * art_scale

        nx = self.base_norms[:, 0]
        ny = self.base_norms[:, 1]
        nz = self.base_norms[:, 2]
        self.cur_norms[:, 0] = nx * cos_t - nz * sin_t
        self.cur_norms[:, 1] = ny
        self.cur_norms[:, 2] = nx * sin_t + nz * cos_t

    def draw(self, shininess=36.0):
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, self.cur_verts)
        glNormalPointer(GL_FLOAT, 0, self.cur_norms)
        glColorPointer(4, GL_FLOAT, 0, self.colors)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, self.indices)

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)


# =============================================================================
# PROCEDURAL 3D ANATOMICAL HEART BUILDER
# =============================================================================

class AnatomicalHeartModel:
    """
    Constructs all solid internal structures and the all-angle translucent outer layers.
    """
    def __init__(self):
        self.papillary_anchors = []
        self._build_all_geometry()

    def _build_outer_epicardial_shell(self):
        n_lat = 52
        n_lon = 68
        u = np.linspace(0.02, math.pi - 0.02, n_lat)
        v = np.linspace(0.0, 2.0 * math.pi, n_lon, endpoint=False)

        vertices = np.zeros((n_lat * n_lon, 3), dtype=np.float32)
        for i, phi in enumerate(u):
            cos_p = math.cos(phi)
            sin_p = math.sin(phi)

            for j, theta in enumerate(v):
                r_base = 1.48 * sin_p * (1.0 + 0.18 * cos_p - 0.12 * (cos_p ** 2))
                iv_groove_ant = -0.14 * math.exp(-((theta - 1.38) / 0.32) ** 2) * (sin_p ** 1.5)
                iv_groove_post = -0.11 * math.exp(-((theta - 4.55) / 0.35) ** 2) * (sin_p ** 1.5)
                av_sulcus = -0.15 * math.exp(-((cos_p - 0.34) / 0.16) ** 2)
                ra_bulge = 0.28 * math.exp(-((cos_p - 0.58) / 0.28) ** 2) * math.exp(-((theta - 2.85) / 0.65) ** 2)
                la_bulge = 0.25 * math.exp(-((cos_p - 0.60) / 0.28) ** 2) * math.exp(-((theta - 5.80) / 0.65) ** 2)
                lv_bulge = 0.22 * math.exp(-((cos_p + 0.18) / 0.55) ** 2) * math.exp(-((theta - 0.35) / 0.85) ** 2)

                r = r_base + iv_groove_ant + iv_groove_post + av_sulcus + ra_bulge + la_bulge + lv_bulge

                x = r * math.cos(theta) * 1.12
                z = r * math.sin(theta) * 0.86
                y = 1.38 * cos_p - 0.35

                apex_factor = max(0.0, (0.55 - cos_p) / 1.55) ** 1.35
                x += 0.62 * apex_factor
                z += 0.28 * apex_factor

                vertices[i * n_lon + j] = [x, y, z]

        indices = []
        for i in range(n_lat - 1):
            for j in range(n_lon):
                jn = (j + 1) % n_lon
                a = i * n_lon + j
                b = (i + 1) * n_lon + j
                c = (i + 1) * n_lon + jn
                d = i * n_lon + jn
                indices.extend([a, b, c, a, c, d])

        top_center = np.mean(vertices[:n_lon], axis=0)
        bot_center = np.mean(vertices[(n_lat - 1) * n_lon:], axis=0)
        top_idx = len(vertices)
        bot_idx = len(vertices) + 1
        vertices = np.vstack([vertices, top_center, bot_center]).astype(np.float32)

        for j in range(n_lon):
            jn = (j + 1) % n_lon
            indices.extend([top_idx, j, jn])
            b_curr = (n_lat - 1) * n_lon + j
            b_next = (n_lat - 1) * n_lon + jn
            indices.extend([bot_idx, b_next, b_curr])

        indices = np.array(indices, dtype=np.uint32)
        normals = compute_vertex_normals(vertices, indices)
        return (vertices, normals, indices, (0.88, 0.22, 0.28, 0.24))

    def _build_interventricular_septum(self):
        centerline = catmull_rom_spline([
            [-0.08,  1.15, -0.18],
            [-0.05,  0.55, -0.05],
            [ 0.04, -0.15,  0.08],
            [ 0.25, -0.95,  0.18],
            [ 0.55, -1.68,  0.26],
        ], num_samples=32)
        radii = np.array([
            0.22, 0.28, 0.34, 0.38, 0.42, 0.44, 0.45, 0.45,
            0.44, 0.43, 0.42, 0.40, 0.38, 0.36, 0.33, 0.29,
            0.24, 0.18
        ], dtype=np.float32)
        radii = np.interp(np.linspace(0, 1, 32), np.linspace(0, 1, len(radii)), radii)
        verts, norms, idx = build_tube_mesh(centerline, radii, radial_segments=24)
        verts[:, 0] *= 0.52
        verts[:, 2] *= 1.65
        norms = compute_vertex_normals(verts, idx)
        return (verts, norms, idx, (0.64, 0.14, 0.18, 1.0))

    def _build_inner_chamber_cradles(self):
        cradles = []

        rv_path = catmull_rom_spline([
            [-0.90,  0.28,  0.12],
            [-0.95, -0.35,  0.18],
            [-0.65, -1.05,  0.22],
            [-0.12, -1.48,  0.28],
            [-0.22, -0.45,  0.42],
            [-0.25,  0.48,  0.48],
        ], num_samples=32)
        rv_radii = np.linspace(0.54, 0.34, 32)
        v_rv, n_rv, i_rv = build_tube_mesh(rv_path, rv_radii, radial_segments=18, arc_range=(2.3, 5.4))
        cradles.append((v_rv, n_rv, i_rv, (0.68, 0.18, 0.22, 1.0)))

        lv_path = catmull_rom_spline([
            [ 0.78,  0.30, -0.12],
            [ 0.92, -0.35,  0.02],
            [ 0.88, -1.15,  0.15],
            [ 0.68, -1.72,  0.28],
            [ 0.38, -0.85,  0.12],
            [ 0.22,  0.42,  0.04],
        ], num_samples=34)
        lv_radii = np.linspace(0.56, 0.36, 34)
        v_lv, n_lv, i_lv = build_tube_mesh(lv_path, lv_radii, radial_segments=20, arc_range=(3.5, 6.7))
        cradles.append((v_lv, n_lv, i_lv, (0.76, 0.16, 0.20, 1.0)))

        ra_path = catmull_rom_spline([
            [-1.12,  0.95, -0.08],
            [-1.18,  0.65, -0.02],
            [-1.00,  0.28,  0.12],
        ], num_samples=18)
        ra_radii = np.interp(np.linspace(0, 1, 18), [0.0, 0.5, 1.0], [0.36, 0.48, 0.42])
        v_ra, n_ra, i_ra = build_tube_mesh(ra_path, ra_radii, radial_segments=16, arc_range=(2.1, 4.9))
        cradles.append((v_ra, n_ra, i_ra, (0.62, 0.20, 0.28, 1.0)))

        la_path = catmull_rom_spline([
            [ 0.86,  0.95, -0.40],
            [ 0.84,  0.68, -0.32],
            [ 0.76,  0.30, -0.15],
        ], num_samples=18)
        la_radii = np.interp(np.linspace(0, 1, 18), [0.0, 0.5, 1.0], [0.35, 0.46, 0.40])
        v_la, n_la, i_la = build_tube_mesh(la_path, la_radii, radial_segments=16, arc_range=(3.5, 6.3))
        cradles.append((v_la, n_la, i_la, (0.78, 0.22, 0.26, 1.0)))

        trabeculae_specs = [
            ([[-0.85, -0.45, 0.02], [-0.55, -0.95, 0.08], [-0.15, -1.25, 0.18]], 0.075),
            ([[-0.72, -0.15, -0.08], [-0.42, -0.68, -0.02], [-0.08, -0.92, 0.12]], 0.068),
            ([[ 0.85, -0.38, -0.18], [ 0.78, -0.98, -0.05], [ 0.55, -1.48, 0.16]], 0.082),
            ([[ 0.55, -0.25, -0.26], [ 0.52, -0.82, -0.14], [ 0.38, -1.32, 0.08]], 0.075),
        ]
        for pts, rad in trabeculae_specs:
            curve = catmull_rom_spline(pts, num_samples=14)
            vt, nt, it = build_tube_mesh(curve, rad, radial_segments=10)
            cradles.append((vt, nt, it, (0.72, 0.15, 0.19, 1.0)))

        return cradles

    def _build_papillary_muscles(self):
        specs = [
            {"base": np.array([-0.92, -0.88,  0.32]), "tip": np.array([-0.82, -0.25,  0.24]), "valve": "tricuspid", "leaf_idx": 0},
            {"base": np.array([-0.68, -1.05, -0.05]), "tip": np.array([-0.74, -0.28,  0.04]), "valve": "tricuspid", "leaf_idx": 1},
            {"base": np.array([-0.22, -0.82,  0.16]), "tip": np.array([-0.48, -0.22,  0.16]), "valve": "tricuspid", "leaf_idx": 2},
            {"base": np.array([ 0.92, -1.05,  0.10]), "tip": np.array([ 0.76, -0.26, -0.04]), "valve": "mitral", "leaf_idx": 0},
            {"base": np.array([ 0.58, -1.12, -0.24]), "tip": np.array([ 0.64, -0.28, -0.22]), "valve": "mitral", "leaf_idx": 1},
        ]
        meshes = []
        self.papillary_anchors = []
        for sp in specs:
            curve = catmull_rom_spline([
                sp["base"],
                0.5 * (sp["base"] + sp["tip"]) + np.array([0.0, 0.04, 0.02]),
                sp["tip"]
            ], num_samples=12)
            radii = np.linspace(0.13, 0.045, 12)
            vp, np_norm, ip = build_tube_mesh(curve, radii, radial_segments=12)
            meshes.append((vp, np_norm, ip, (0.75, 0.18, 0.22, 1.0)))
            self.papillary_anchors.append(sp)
        return meshes

    def _build_translucent_vessels(self):
        vessels = []
        artery_rgba = (0.95, 0.20, 0.24, 0.18)
        vein_rgba = (0.24, 0.45, 0.95, 0.18)
        pulm_art_rgba = (0.36, 0.46, 0.92, 0.18)
        pulm_vein_rgba = (0.92, 0.24, 0.30, 0.18)

        # 1. Ascending Aorta, Aortic Arch, Descending Aorta
        aorta_pts = [
            [ 0.20,  0.62,  0.04],
            [ 0.26,  1.18,  0.14],
            [ 0.15,  1.82,  0.10],
            [-0.12,  2.18, -0.12],
            [-0.42,  2.02, -0.48],
            [-0.48,  1.35, -0.78],
            [-0.42,  0.10, -0.88],
            [-0.38, -1.55, -0.85],
        ]
        aorta_curve = catmull_rom_spline(aorta_pts, num_samples=52)
        aorta_radii = np.interp(np.linspace(0, 1, 52), [0.0, 0.25, 0.5, 1.0], [0.34, 0.38, 0.34, 0.28])
        vo, no, io = build_tube_mesh(aorta_curve, aorta_radii, radial_segments=22)
        vessels.append((vo, no, io, artery_rgba))

        branches = [
            ([[ 0.10, 2.02,  0.04], [ 0.22, 2.48,  0.06], [ 0.32, 2.85,  0.08]], 0.13),
            ([[-0.08, 2.15, -0.08], [-0.06, 2.58, -0.08], [-0.04, 2.92, -0.08]], 0.10),
            ([[-0.28, 2.12, -0.28], [-0.35, 2.55, -0.30], [-0.42, 2.88, -0.32]], 0.10),
        ]
        for b_pts, b_rad in branches:
            bc = catmull_rom_spline(b_pts, num_samples=14)
            vb, nb, ib = build_tube_mesh(bc, b_rad, radial_segments=14)
            vessels.append((vb, nb, ib, artery_rgba))

        # 2. Pulmonary Artery Trunk & Left/Right Pulmonary Arteries
        pulm_trunk_pts = [
            [-0.25,  0.88,  0.52],
            [-0.28,  1.35,  0.42],
            [-0.18,  1.68,  0.18],
            [ 0.22,  1.72, -0.15],
            [ 0.95,  1.68, -0.32],
            [ 1.48,  1.62, -0.42],
        ]
        pulm_curve = catmull_rom_spline(pulm_trunk_pts, num_samples=38)
        pulm_radii = np.interp(np.linspace(0, 1, 38), [0.0, 0.45, 1.0], [0.32, 0.31, 0.22])
        vp, np_n, ip = build_tube_mesh(pulm_curve, pulm_radii, radial_segments=20)
        vessels.append((vp, np_n, ip, pulm_art_rgba))

        rpa_pts = [
            [-0.18, 1.65,  0.15],
            [-0.68, 1.62, -0.12],
            [-1.38, 1.56, -0.28],
        ]
        rpa_curve = catmull_rom_spline(rpa_pts, num_samples=20)
        vrpa, nrpa, irpa = build_tube_mesh(rpa_curve, 0.21, radial_segments=16)
        vessels.append((vrpa, nrpa, irpa, pulm_art_rgba))

        # 3. Superior Vena Cava (SVC) & Inferior Vena Cava (IVC)
        svc_curve = catmull_rom_spline([
            [-1.12,  2.65, -0.15],
            [-1.12,  1.85, -0.10],
            [-1.12,  1.05, -0.02],
        ], num_samples=24)
        v_svc, n_svc, i_svc = build_tube_mesh(svc_curve, 0.28, radial_segments=20)
        vessels.append((v_svc, n_svc, i_svc, vein_rgba))

        ivc_curve = catmull_rom_spline([
            [-1.12, -1.75, -0.35],
            [-1.12, -0.65, -0.22],
            [-1.10,  0.48, -0.05],
        ], num_samples=24)
        v_ivc, n_ivc, i_ivc = build_tube_mesh(ivc_curve, 0.30, radial_segments=20)
        vessels.append((v_ivc, n_ivc, i_ivc, vein_rgba))

        # 4. Four Pulmonary Veins entering Left Atrium
        pv_specs = [
            [[ 1.85, 1.25, -0.45], [ 1.35, 1.05, -0.42], [ 0.92, 0.92, -0.38]],
            [[ 1.80, 0.65, -0.48], [ 1.32, 0.72, -0.42], [ 0.90, 0.75, -0.38]],
            [[-0.35, 1.15, -0.65], [ 0.25, 1.02, -0.55], [ 0.75, 0.92, -0.42]],
            [[-0.30, 0.68, -0.68], [ 0.25, 0.72, -0.56], [ 0.75, 0.74, -0.42]],
        ]
        for pv_pts in pv_specs:
            pvc = catmull_rom_spline(pv_pts, num_samples=16)
            vpv, npv, ipv = build_tube_mesh(pvc, 0.15, radial_segments=14)
            vessels.append((vpv, npv, ipv, pulm_vein_rgba))

        # 5. Translucent Epicardial Coronary Arteries & Great Cardiac Vein
        cor_artery_rgba = (0.95, 0.16, 0.20, 0.28)
        cor_vein_rgba = (0.20, 0.42, 0.95, 0.28)

        cor_specs = [
            ([[ 0.28, 0.68, 0.18], [ 0.12, 0.38, 0.68], [ 0.16, -0.25, 0.86], [ 0.34, -0.98, 0.76], [ 0.58, -1.62, 0.52]], 0.048, 0.020, cor_artery_rgba),
            ([[ 0.16, -0.15, 0.85], [ 0.62, -0.52, 0.78], [ 0.95, -1.02, 0.52]], 0.035, 0.016, cor_artery_rgba),
            ([[ 0.28, 0.68, 0.18], [ 0.85, 0.42, 0.32], [ 1.25, 0.18, -0.15], [ 1.05, -0.35, -0.58]], 0.044, 0.018, cor_artery_rgba),
            ([[ 0.12, 0.65, 0.22], [-0.55, 0.35, 0.62], [-1.12, -0.05, 0.38], [-1.05, -0.52, -0.25], [-0.45, -0.95, -0.58]], 0.046, 0.020, cor_artery_rgba),
            ([[ 0.48, -1.52, 0.56], [ 0.24, -0.88, 0.80], [ 0.06, -0.18, 0.88], [ 0.72, 0.35, 0.42], [ 1.18, 0.22, -0.22]], 0.020, 0.042, cor_vein_rgba),
        ]
        for pts, r0, r1, col in cor_specs:
            cc = catmull_rom_spline(pts, num_samples=24)
            cr = np.linspace(r0, r1, 24)
            vc, nc, ic = build_tube_mesh(cc, cr, radial_segments=10)
            vessels.append((vc, nc, ic, col))

        return vessels

    def _build_all_geometry(self):
        solid_list = [self._build_interventricular_septum()]
        solid_list.extend(self._build_inner_chamber_cradles())
        solid_list.extend(self._build_papillary_muscles())
        self.solid_batch = FastDeformableBatch(solid_list)

        self.shell_batch = FastDeformableBatch([self._build_outer_epicardial_shell()])
        self.vessel_batch = FastDeformableBatch(self._build_translucent_vessels())


# =============================================================================
# 4 MULTI-LEAFLET HEART VALVES & DYNAMIC CHORDAE TENDINEAE
# =============================================================================

class HeartValveSystem:
    """Renders and animates all 4 cardiac valves and dynamic chordae tendineae."""
    def __init__(self):
        self.valves = {
            "tricuspid": {
                "pos": np.array([-0.78,  0.22,  0.18], dtype=np.float32),
                "normal": normalize(np.array([0.12, -0.96, 0.14], dtype=np.float32)),
                "radius": 0.34,
                "num_leaflets": 3,
            },
            "mitral": {
                "pos": np.array([ 0.72,  0.26, -0.12], dtype=np.float32),
                "normal": normalize(np.array([0.06, -0.96, 0.16], dtype=np.float32)),
                "radius": 0.32,
                "num_leaflets": 2,
            },
            "pulmonary": {
                "pos": np.array([-0.25,  0.86,  0.50], dtype=np.float32),
                "normal": normalize(np.array([-0.05, 0.94, -0.22], dtype=np.float32)),
                "radius": 0.25,
                "num_leaflets": 3,
            },
            "aortic": {
                "pos": np.array([ 0.20,  0.60,  0.04], dtype=np.float32),
                "normal": normalize(np.array([0.10, 0.96, 0.16], dtype=np.float32)),
                "radius": 0.26,
                "num_leaflets": 3,
            },
        }
        self._build_unit_annulus_list()

    def _build_unit_annulus_list(self):
        self.annulus_list = glGenLists(1)
        glNewList(self.annulus_list, GL_COMPILE)
        n_ring = 24
        r_tube = 0.11
        for i in range(n_ring):
            t0 = (2.0 * math.pi * i) / n_ring
            t1 = (2.0 * math.pi * (i + 1)) / n_ring
            glBegin(GL_QUAD_STRIP)
            for k in range(9):
                phi = (2.0 * math.pi * k) / 8
                for t_ang in (t0, t1):
                    rx = math.cos(t_ang)
                    ry = math.sin(t_ang)
                    nx = math.cos(phi) * rx
                    ny = math.cos(phi) * ry
                    nz = math.sin(phi)
                    px = (1.0 + r_tube * math.cos(phi)) * rx
                    py = (1.0 + r_tube * math.cos(phi)) * ry
                    pz = r_tube * math.sin(phi)
                    glNormal3f(nx, ny, nz)
                    glVertex3f(px, py, pz)
            glEnd()
        glEndList()

    def render_valves_and_chordae(self, cycle: CardiacCycleEngine, papillary_anchors):
        open_states = {
            "tricuspid": cycle.tricuspid_open,
            "mitral": cycle.mitral_open,
            "pulmonary": cycle.pulmonary_open,
            "aortic": cycle.aortic_open,
        }

        leaflet_tips_world = {"tricuspid": [], "mitral": []}

        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.55, 0.55, 0.55, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 48.0)

        v_names = list(self.valves.keys())
        centers_base = np.array([self.valves[k]["pos"] for k in v_names], dtype=np.float32)
        centers_def = deform_points_direct(centers_base, cycle)

        for idx_v, v_name in enumerate(v_names):
            cfg = self.valves[v_name]
            c_def = centers_def[idx_v]
            n_vec = cfg["normal"]
            up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if abs(np.dot(n_vec, up_ref)) > 0.9:
                up_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            u_vec = normalize(np.cross(n_vec, up_ref))
            v_vec = normalize(np.cross(n_vec, u_vec))

            rad = cfg["radius"]
            n_leaf = cfg["num_leaflets"]
            open_amt = open_states[v_name]

            glColor3f(0.88, 0.82, 0.74)
            mat = np.identity(4, dtype=np.float32)
            mat[:3, 0] = u_vec * rad
            mat[:3, 1] = v_vec * rad
            mat[:3, 2] = n_vec * rad
            mat[:3, 3] = c_def
            glPushMatrix()
            glMultMatrixf(mat.T)
            glCallList(self.annulus_list)
            glPopMatrix()

            glColor3f(0.96, 0.93, 0.86)
            for l_idx in range(n_leaf):
                ang_start = (2.0 * math.pi * l_idx) / n_leaf + 0.04
                ang_end = (2.0 * math.pi * (l_idx + 1)) / n_leaf - 0.04
                ang_mid = 0.5 * (ang_start + ang_end)

                arcs = np.linspace(ang_start, ang_end, 9)
                tip_radial_dist = rad * (0.04 + 0.82 * open_amt)
                tip_axial_drop = rad * (0.14 + 0.78 * open_amt)

                mid_radial = math.cos(ang_mid) * u_vec + math.sin(ang_mid) * v_vec
                tip_world = c_def + tip_radial_dist * mid_radial + tip_axial_drop * n_vec

                if v_name in leaflet_tips_world:
                    leaflet_tips_world[v_name].append(tip_world)

                glBegin(GL_TRIANGLE_STRIP)
                for a in arcs:
                    rim_dir = math.cos(a) * u_vec + math.sin(a) * v_vec
                    rim_pt = c_def + rad * rim_dir

                    angular_center_weight = math.cos((a - ang_mid) * (n_leaf / 2.0))
                    inner_r = rad * (1.0 - (0.96 - 0.80 * open_amt) * angular_center_weight)
                    inner_z = tip_axial_drop * angular_center_weight
                    inner_pt = c_def + inner_r * rim_dir + inner_z * n_vec

                    leaf_norm = normalize(n_vec - 0.55 * rim_dir * open_amt)
                    glNormal3fv(leaf_norm)
                    glVertex3fv(rim_pt)
                    glVertex3fv(inner_pt)
                glEnd()

        glColor3f(0.92, 0.87, 0.78)
        glLineWidth(2.0)
        pap_tips_base = np.array([a["tip"] for a in papillary_anchors], dtype=np.float32)
        pap_tips_def = deform_points_direct(pap_tips_base, cycle)

        t_vals = np.linspace(0.0, 1.0, 7, dtype=np.float32)[:, None]
        glBegin(GL_LINES)
        glNormal3f(0.0, 1.0, 0.0)
        for idx_a, anchor in enumerate(papillary_anchors):
            v_type = anchor["valve"]
            l_idx = anchor["leaf_idx"]
            tips = leaflet_tips_world.get(v_type, [])
            if l_idx < len(tips):
                pap_tip = pap_tips_def[idx_a]
                leaf_tip = tips[l_idx]
                for offset_scale in (-0.045, 0.0, 0.045):
                    cord_end = leaf_tip + np.array([offset_scale, 0.0, -offset_scale], dtype=np.float32)
                    mid_pt = 0.5 * (pap_tip + cord_end) + np.array([0.0, 0.02, 0.02], dtype=np.float32)
                    pts_c = ((1.0 - t_vals) ** 2) * pap_tip + 2.0 * (1.0 - t_vals) * t_vals * mid_pt + (t_vals ** 2) * cord_end
                    for k in range(len(pts_c) - 1):
                        glVertex3fv(pts_c[k])
                        glVertex3fv(pts_c[k + 1])
        glEnd()


# =============================================================================
# VECTORIZED HEMODYNAMIC BLOOD FLOW PARTICLE SYSTEM
# =============================================================================

class BloodFlowSystem:
    """
    Fully vectorized 3D erythrocyte particle system (380 particles).
    """
    def __init__(self, num_particles=380):
        self.num_particles = num_particles
        self._build_hemodynamic_splines()
        self._build_batched_rbc_template()
        self.reset_particles()

    def _build_hemodynamic_splines(self):
        p0 = [
            [-1.12,  2.55, -0.15],
            [-1.12,  1.65, -0.08],
            [-1.12,  0.82,  0.02],
            [-0.78,  0.22,  0.18],
            [-0.72, -0.58,  0.24],
            [-0.25, -1.25,  0.30],
            [-0.24, -0.15,  0.46],
            [-0.25,  0.86,  0.50],
            [-0.24,  1.52,  0.32],
            [ 0.35,  1.70, -0.18],
            [ 1.42,  1.62, -0.40],
        ]
        p1 = [
            [-1.12, -1.65, -0.35],
            [-1.12, -0.45, -0.18],
            [-1.08,  0.68,  0.04],
            [-0.78,  0.22,  0.18],
            [-0.82, -0.65,  0.20],
            [-0.32, -1.32,  0.28],
            [-0.26, -0.10,  0.48],
            [-0.25,  0.86,  0.50],
            [-0.22,  1.55,  0.25],
            [-0.72,  1.62, -0.12],
            [-1.35,  1.56, -0.28],
        ]
        p2 = [
            [ 1.78,  1.15, -0.44],
            [ 1.25,  0.96, -0.40],
            [ 0.84,  0.76, -0.32],
            [ 0.72,  0.26, -0.12],
            [ 0.82, -0.58,  0.06],
            [ 0.66, -1.55,  0.24],
            [ 0.36, -0.42,  0.10],
            [ 0.20,  0.60,  0.04],
            [ 0.24,  1.45,  0.12],
            [-0.12,  2.16, -0.12],
            [-0.46,  1.45, -0.75],
            [-0.38, -1.45, -0.85],
        ]
        p3 = [
            [-0.28,  0.95, -0.65],
            [ 0.32,  0.88, -0.52],
            [ 0.80,  0.72, -0.32],
            [ 0.72,  0.26, -0.12],
            [ 0.76, -0.62,  0.02],
            [ 0.60, -1.52,  0.22],
            [ 0.34, -0.38,  0.08],
            [ 0.20,  0.60,  0.04],
            [ 0.22,  1.52,  0.10],
            [-0.06,  2.14, -0.08],
            [-0.05,  2.85, -0.08],
        ]

        self.n_samples = 220
        self.splines = np.stack([
            catmull_rom_spline(p0, num_samples=self.n_samples),
            catmull_rom_spline(p1, num_samples=self.n_samples),
            catmull_rom_spline(p2, num_samples=self.n_samples),
            catmull_rom_spline(p3, num_samples=self.n_samples),
        ], axis=0).astype(np.float32)

        self.tangents = np.stack([
            normalize(np.gradient(self.splines[k], axis=0)) for k in range(4)
        ], axis=0).astype(np.float32)

        self.frame_n1 = np.zeros_like(self.tangents)
        self.frame_n2 = np.zeros_like(self.tangents)
        for k in range(4):
            t_arr = self.tangents[k]
            up = np.zeros_like(t_arr)
            up[:, 2] = 1.0
            parallel = np.abs(t_arr[:, 2]) > 0.9
            up[parallel, 2] = 0.0
            up[parallel, 0] = 1.0
            n1 = normalize(np.cross(t_arr, up))
            n2 = normalize(np.cross(t_arr, n1))
            self.frame_n1[k] = n1
            self.frame_n2[k] = n2

    def _build_batched_rbc_template(self):
        n_lat, n_lon = 6, 10
        radius = 0.046
        verts = []
        norms = []
        for i in range(n_lat + 1):
            u = math.pi * i / n_lat
            r_xy = math.sin(u)
            z_prof = 0.42 * math.cos(u) * (0.35 + 0.65 * (r_xy ** 2))
            for j in range(n_lon):
                v = 2.0 * math.pi * j / n_lon
                x = radius * r_xy * math.cos(v)
                y = radius * r_xy * math.sin(v)
                z = radius * z_prof
                verts.append([x, y, z])
                norms.append(normalize(np.array([x, y, z * 2.2], dtype=np.float32)))

        verts = np.array(verts, dtype=np.float32)
        norms = np.array(norms, dtype=np.float32)

        indices = []
        for i in range(n_lat):
            for j in range(n_lon):
                jn = (j + 1) % n_lon
                a = i * n_lon + j
                b = (i + 1) * n_lon + j
                c = (i + 1) * n_lon + jn
                d = i * n_lon + jn
                indices.extend([a, b, c, a, c, d])
        indices = np.array(indices, dtype=np.uint32)

        self.rbc_base_verts = verts
        self.rbc_base_norms = norms
        n_v = len(verts)

        self.batched_norms = np.tile(norms, (self.num_particles, 1)).astype(np.float32)
        offsets = (np.arange(self.num_particles, dtype=np.uint32) * n_v)[:, None]
        self.batched_indices = (indices[None, :] + offsets).ravel()
        self.batched_verts = np.zeros((self.num_particles, n_v, 3), dtype=np.float32)

    def reset_particles(self):
        rng = np.random.default_rng(42)
        self.path_id = rng.integers(0, 4, size=self.num_particles)
        self.s = rng.uniform(0.0, 1.0, size=self.num_particles).astype(np.float32)
        self.radial_angle = rng.uniform(0.0, 2.0 * math.pi, size=self.num_particles).astype(np.float32)
        self.radial_dist = rng.uniform(0.01, 0.095, size=self.num_particles).astype(np.float32)
        self.speed_mult = rng.uniform(0.86, 1.16, size=self.num_particles).astype(np.float32)
        self.current_vel = np.full(self.num_particles, 0.25, dtype=np.float32)
        self.is_right = np.isin(self.path_id, (0, 1))

        n_v = len(self.rbc_base_verts)
        self.batched_colors = np.zeros((self.num_particles, n_v, 3), dtype=np.float32)
        self.batched_colors[self.is_right] = [0.14, 0.46, 0.98]
        self.batched_colors[~self.is_right] = [0.98, 0.14, 0.18]
        self.batched_colors = self.batched_colors.reshape(-1, 3)

        self.streak_colors = np.zeros((self.num_particles, 2, 3), dtype=np.float32)
        self.streak_colors[self.is_right, 0] = [0.25, 0.58, 1.0]
        self.streak_colors[self.is_right, 1] = [0.08, 0.20, 0.38]
        self.streak_colors[~self.is_right, 0] = [1.0, 0.30, 0.34]
        self.streak_colors[~self.is_right, 1] = [0.38, 0.10, 0.12]
        self.streak_colors = self.streak_colors.reshape(-1, 3)
        self.streak_verts = np.zeros((self.num_particles, 2, 3), dtype=np.float32)

    def update(self, dt, cycle: CardiacCycleEngine):
        if dt <= 0.0:
            return

        av_gate_s = 0.30
        sl_gate_s = 0.64

        av_open = np.where(self.is_right, cycle.tricuspid_open, cycle.mitral_open)
        sl_open = np.where(self.is_right, cycle.pulmonary_open, cycle.aortic_open)
        s = self.s

        vel = np.empty_like(s)
        swirl = np.full_like(s, 1.8)

        m_venous = s < 0.20
        m_atrium = (s >= 0.20) & (s < av_gate_s)
        m_vent_in = (s >= av_gate_s) & (s < 0.48)
        m_vent_out = (s >= 0.48) & (s < sl_gate_s)
        m_artery = s >= sl_gate_s

        vel[m_venous] = 0.28 + 0.14 * av_open[m_venous]

        atrium_open = av_open > 0.15
        vel_atrium = np.where(
            atrium_open,
            0.32 + 0.65 * max(0.0, cycle.atrial_contraction) + 0.22 * av_open,
            0.05
        )
        vel[m_atrium] = vel_atrium[m_atrium]
        swirl[m_atrium & (~atrium_open)] += 4.5

        vel[m_vent_in] = 0.38 * av_open[m_vent_in] + 0.22

        vent_open = sl_open > 0.15
        vel_vent_out = np.where(
            vent_open,
            0.45 + 0.95 * cycle.ventricular_contraction,
            0.06
        )
        vel[m_vent_out] = vel_vent_out[m_vent_out]
        swirl[m_vent_out & (~vent_open)] += 5.0

        vel[m_artery] = 0.24 + 0.98 * cycle.arterial_pulse

        vel *= self.speed_mult
        s_next = s + vel * dt

        block_av = (s < av_gate_s) & (s_next >= av_gate_s) & (av_open < 0.20)
        s_next[block_av] = av_gate_s - 0.004
        vel[block_av] = 0.02

        block_sl = (s < sl_gate_s) & (s_next >= sl_gate_s) & (sl_open < 0.20)
        s_next[block_sl] = sl_gate_s - 0.004
        vel[block_sl] = 0.02

        self.s = np.mod(s_next, 1.0)
        self.current_vel = vel
        self.radial_angle += swirl * dt

    def render(self, cycle: CardiacCycleEngine):
        idx_f = self.s * (self.n_samples - 1)
        idx0 = idx_f.astype(np.int32)
        idx1 = np.minimum(idx0 + 1, self.n_samples - 1)
        frac = (idx_f - idx0)[:, None]

        p_ids = self.path_id
        base_pts = (1.0 - frac) * self.splines[p_ids, idx0] + frac * self.splines[p_ids, idx1]
        tangents = normalize((1.0 - frac) * self.tangents[p_ids, idx0] + frac * self.tangents[p_ids, idx1])
        n1 = self.frame_n1[p_ids, idx0]
        n2 = self.frame_n2[p_ids, idx0]

        cos_a = np.cos(self.radial_angle)[:, None]
        sin_a = np.sin(self.radial_angle)[:, None]
        dist = self.radial_dist[:, None]

        world_pts = base_pts + dist * (cos_a * n1 + sin_a * n2)
        world_pts = deform_points_direct(world_pts, cycle)

        self.batched_verts[:] = world_pts[:, None, :] + self.rbc_base_verts[None, :, :]
        flat_verts = self.batched_verts.reshape(-1, 3)

        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.65, 0.65, 0.65, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 55.0)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, flat_verts)
        glNormalPointer(GL_FLOAT, 0, self.batched_norms)
        glColorPointer(3, GL_FLOAT, 0, self.batched_colors)
        glDrawElements(GL_TRIANGLES, len(self.batched_indices), GL_UNSIGNED_INT, self.batched_indices)

        glDisableClientState(GL_NORMAL_ARRAY)

        streak_len = np.minimum(0.16, 0.14 * self.current_vel)[:, None]
        self.streak_verts[:, 0, :] = world_pts
        self.streak_verts[:, 1, :] = world_pts - streak_len * tangents

        glDisable(GL_LIGHTING)
        glLineWidth(1.8)
        glVertexPointer(3, GL_FLOAT, 0, self.streak_verts.reshape(-1, 3))
        glColorPointer(3, GL_FLOAT, 0, self.streak_colors)
        glDrawArrays(GL_LINES, 0, self.num_particles * 2)
        glEnable(GL_LIGHTING)

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)


# =============================================================================
# 2D HUD OVERLAY, SPEED SLIDER, PCG WAVEFORM & AUDIO BUTTON MANAGER
# =============================================================================

class HUDOverlay:
    """Renders crisp 2D text panels, live ECG + PCG heart sound waveforms, and interactive controls."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        pygame.font.init()
        self.font_title = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_body = pygame.font.SysFont("Consolas", 13, bold=False)
        self.font_btn = pygame.font.SysFont("Consolas", 14, bold=True)
        self.tex_cache = {}

        # Bottom Control Bar Interactive Rectangles
        y_btn = height - 54
        self.btn_pause_rect = pygame.Rect(18, y_btn, 148, 38)
        self.btn_reset_rect = pygame.Rect(176, y_btn, 148, 38)
        self.btn_slower_rect = pygame.Rect(338, y_btn, 42, 38)
        self.slider_rect = pygame.Rect(388, y_btn, 212, 38)
        self.btn_faster_rect = pygame.Rect(608, y_btn, 42, 38)
        self.btn_mute_rect = pygame.Rect(664, y_btn, 175, 38)

    def _get_text_texture(self, text, font, color):
        key = (text, id(font), color)
        if key in self.tex_cache:
            return self.tex_cache[key]

        surf = font.render(text, True, color)
        tex_data = pygame.image.tostring(surf, "RGBA", True)
        w, h = surf.get_size()

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)

        if len(self.tex_cache) > 180:
            for old_id, _, _ in self.tex_cache.values():
                glDeleteTextures([old_id])
            self.tex_cache.clear()

        self.tex_cache[key] = (tex_id, w, h)
        return tex_id, w, h

    def draw_text(self, x, y, text, font=None, color=(235, 242, 250)):
        if font is None:
            font = self.font_body
        tex_id, w, h = self._get_text_texture(text, font, color)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(x, y)
        glTexCoord2f(1, 1); glVertex2f(x + w, y)
        glTexCoord2f(1, 0); glVertex2f(x + w, y + h)
        glTexCoord2f(0, 0); glVertex2f(x, y + h)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def draw_box(self, x, y, w, h, fill_rgba, border_rgba=None):
        glColor4fv(fill_rgba)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()
        if border_rgba is not None:
            glColor4fv(border_rgba)
            glLineWidth(1.5)
            glBegin(GL_LINE_LOOP)
            glVertex2f(x, y)
            glVertex2f(x + w, y)
            glVertex2f(x + w, y + h)
            glVertex2f(x, y + h)
            glEnd()

    def speed_from_mouse_x(self, mx):
        pad = 12
        t = (mx - (self.slider_rect.x + pad)) / float(self.slider_rect.w - 2 * pad)
        t = max(0.0, min(1.0, t))
        return round(0.10 + t * 2.90, 2)

    def render(self, cycle: CardiacCycleEngine, audio: CardiacAudioEngine, cam_rot, sim_speed, paused, fps, mouse_pos):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # 1. Left Telemetry & Controls Panel
        self.draw_box(16, 16, 448, 294, (0.05, 0.08, 0.14, 0.78), (0.28, 0.45, 0.68, 0.85))
        self.draw_text(28, 25, "3D ANATOMICAL HEART & HEMODYNAMICS", self.font_title, (255, 110, 125))
        self.draw_text(28, 48, f"Cardiac Phase : {cycle.phase_name}", self.font_body, (230, 240, 255))
        eff_bpm = cycle.bpm * sim_speed
        self.draw_text(
            28, 67,
            f"Sim Speed     : {sim_speed:.2f}x  (Effective: {eff_bpm:.0f} BPM | {fps:.0f} FPS)",
            self.font_body, (130, 240, 255)
        )

        # Live Auscultation Status
        snd_status = "MUTED" if not audio.enabled else audio.active_sound_label
        snd_col = (255, 130, 130) if not audio.enabled else (255, 230, 120)
        self.draw_text(28, 86, f"Stethoscope   : {snd_status}", self.font_body, snd_col)

        rx, ry, rz = cam_rot
        self.draw_text(28, 106, f"Camera Angles : X={rx:5.1f}° | Y={ry:5.1f}° | Z={rz:5.1f}°", self.font_body, (160, 250, 195))
        self.draw_text(28, 129, "Keyboard & Mouse Controls:", self.font_body, (255, 215, 130))
        self.draw_text(28, 148, "  [UP / DOWN]          : Rotate X-Axis (Pitch)", self.font_body, (210, 220, 235))
        self.draw_text(28, 166, "  [LEFT / RIGHT]       : Rotate Y-Axis (Yaw)", self.font_body, (210, 220, 235))
        self.draw_text(28, 184, "  [SHIFT+LEFT / RIGHT] : Rotate Z-Axis (Roll) [or Q/E]", self.font_body, (210, 220, 235))
        self.draw_text(28, 202, "  [W / S] or Scroll    : Zoom Camera In / Out", self.font_body, (210, 220, 235))
        self.draw_text(28, 220, "  [ [ / ] ] or [1 - 5] : Adjust Simulation Speed (0.1x-3.0x)", self.font_body, (155, 235, 255))
        self.draw_text(28, 238, "  [P] Pause | [R] Reset | [M] Mute/Unmute Heart Sounds", self.font_body, (255, 225, 155))
        self.draw_text(28, 260, "Legend: [Blue] Deoxygenated (RA/RV) | [Red] Oxygenated (LA/LV)", self.font_body, (195, 210, 245))
        self.draw_text(28, 279, "Vessels: 360° All-Angle Translucent Veins & Arteries", self.font_body, (175, 225, 205))

        # 2. Right Panel: Live Valve Status, Lead II ECG & Phonocardiogram (PCG)
        rx_box = self.width - 340
        self.draw_box(rx_box, 16, 324, 304, (0.05, 0.08, 0.14, 0.78), (0.28, 0.45, 0.68, 0.85))
        self.draw_text(rx_box + 14, 24, "VALVE DYNAMICS, ECG & PCG", self.font_title, (130, 230, 255))

        valves_info = [
            ("Tricuspid (RA->RV)", cycle.tricuspid_open),
            ("Mitral    (LA->LV)", cycle.mitral_open),
            ("Pulmonary (RV->PA)", cycle.pulmonary_open),
            ("Aortic    (LV->AO)", cycle.aortic_open),
        ]
        for idx, (v_label, v_open) in enumerate(valves_info):
            vy = 54 + idx * 24
            state_txt = "OPEN  " if v_open > 0.25 else "CLOSED"
            col = (110, 250, 145) if v_open > 0.25 else (255, 115, 115)
            self.draw_text(rx_box + 14, vy, f"{v_label}: {state_txt}", self.font_body, col)
            bar_x = rx_box + 225
            self.draw_box(bar_x, vy + 3, 82, 12, (0.18, 0.22, 0.28, 0.9))
            fill_col = (0.25, 0.92, 0.48, 0.95) if v_open > 0.25 else (0.92, 0.28, 0.28, 0.85)
            self.draw_box(bar_x, vy + 3, max(3.0, 82.0 * v_open), 12, fill_col)

        # Live Lead II ECG Waveform Plot
        ecg_x, ecg_y, ecg_w, ecg_h = rx_box + 14, 156, 296, 70
        self.draw_box(ecg_x, ecg_y, ecg_w, ecg_h, (0.02, 0.05, 0.08, 0.92), (0.18, 0.38, 0.42, 0.8))
        self.draw_text(ecg_x + 6, ecg_y + 4, "Lead II ECG (P-QRS-T)", self.font_body, (110, 245, 165))
        glColor4f(0.20, 0.98, 0.48, 1.0)
        glLineWidth(2.0)
        glBegin(GL_LINE_STRIP)
        n_pts = len(cycle.ecg_history)
        for i, val in enumerate(cycle.ecg_history):
            px = ecg_x + (i / max(1, n_pts - 1)) * ecg_w
            py = (ecg_y + ecg_h * 0.68) - val * (ecg_h * 0.46)
            glVertex2f(px, py)
        glEnd()

        # Live Phonocardiogram (PCG - Heart Sound S1/S2) Waveform Plot
        pcg_x, pcg_y, pcg_w, pcg_h = rx_box + 14, 236, 296, 70
        self.draw_box(pcg_x, pcg_y, pcg_w, pcg_h, (0.02, 0.05, 0.08, 0.92), (0.42, 0.35, 0.18, 0.8))
        self.draw_text(pcg_x + 6, pcg_y + 4, "Auscultation PCG (S1 Lub / S2 Dub)", self.font_body, (255, 220, 120))
        glColor4f(1.0, 0.78, 0.25, 1.0)
        glLineWidth(1.8)
        glBegin(GL_LINE_STRIP)
        n_pcg = len(cycle.pcg_history)
        for i, val in enumerate(cycle.pcg_history):
            px = pcg_x + (i / max(1, n_pcg - 1)) * pcg_w
            py = (pcg_y + pcg_h * 0.56) - val * (pcg_h * 0.38)
            glVertex2f(px, py)
        glEnd()

        # 3. Bottom Bar: [P] PAUSE, [R] RESET, Speed Slider, and [M] SOUND Toggle
        p_hover = self.btn_pause_rect.collidepoint(mouse_pos)
        r_hover = self.btn_reset_rect.collidepoint(mouse_pos)
        slow_hover = self.btn_slower_rect.collidepoint(mouse_pos)
        fast_hover = self.btn_faster_rect.collidepoint(mouse_pos)
        mute_hover = self.btn_mute_rect.collidepoint(mouse_pos)

        p_fill = (0.85, 0.48, 0.14, 0.92) if paused else ((0.18, 0.36, 0.58, 0.92) if p_hover else (0.10, 0.22, 0.38, 0.86))
        r_fill = (0.62, 0.18, 0.24, 0.92) if r_hover else (0.10, 0.22, 0.38, 0.86)
        s_fill = (0.18, 0.36, 0.58, 0.92) if slow_hover else (0.10, 0.22, 0.38, 0.86)
        f_fill = (0.18, 0.36, 0.58, 0.92) if fast_hover else (0.10, 0.22, 0.38, 0.86)
        m_fill = (0.16, 0.48, 0.32, 0.92) if audio.enabled else (0.55, 0.20, 0.22, 0.92)
        if mute_hover:
            m_fill = (min(1.0, m_fill[0] + 0.12), min(1.0, m_fill[1] + 0.12), min(1.0, m_fill[2] + 0.12), 0.95)

        self.draw_box(*self.btn_pause_rect, p_fill, (0.55, 0.78, 1.0, 0.95))
        p_label = "[P] RESUME" if paused else "[P] PAUSE"
        self.draw_text(self.btn_pause_rect.x + 26, self.btn_pause_rect.y + 11, p_label, self.font_btn, (255, 255, 255))

        self.draw_box(*self.btn_reset_rect, r_fill, (0.55, 0.78, 1.0, 0.95))
        self.draw_text(self.btn_reset_rect.x + 28, self.btn_reset_rect.y + 11, "[R] RESET", self.font_btn, (255, 255, 255))

        self.draw_box(*self.btn_slower_rect, s_fill, (0.55, 0.78, 1.0, 0.95))
        self.draw_text(self.btn_slower_rect.x + 9, self.btn_slower_rect.y + 11, "[-]", self.font_btn, (255, 255, 255))

        self.draw_box(*self.slider_rect, (0.06, 0.13, 0.24, 0.90), (0.45, 0.70, 0.95, 0.95))
        pad = 12
        track_x = self.slider_rect.x + pad
        track_w = self.slider_rect.w - 2 * pad
        track_y = self.slider_rect.y + 24
        self.draw_box(track_x, track_y, track_w, 6, (0.18, 0.25, 0.36, 1.0))

        t_norm = max(0.0, min(1.0, (sim_speed - 0.10) / 2.90))
        fill_w = max(4.0, track_w * t_norm)
        self.draw_box(track_x, track_y, fill_w, 6, (0.25, 0.85, 0.98, 1.0))
        knob_x = track_x + fill_w - 5
        self.draw_box(knob_x, track_y - 4, 10, 14, (0.95, 0.98, 1.0, 1.0), (0.20, 0.55, 0.85, 1.0))

        self.draw_text(
            self.slider_rect.x + 40,
            self.slider_rect.y + 5,
            f"SPEED: {sim_speed:.2f}x",
            self.font_body,
            (165, 240, 255)
        )

        self.draw_box(*self.btn_faster_rect, f_fill, (0.55, 0.78, 1.0, 0.95))
        self.draw_text(self.btn_faster_rect.x + 9, self.btn_faster_rect.y + 11, "[+]", self.font_btn, (255, 255, 255))

        # Audio Mute / Unmute Button
        self.draw_box(*self.btn_mute_rect, m_fill, (0.55, 0.78, 1.0, 0.95))
        m_label = "[M] SOUND: ON" if audio.enabled else "[M] SOUND: OFF"
        self.draw_text(self.btn_mute_rect.x + 22, self.btn_mute_rect.y + 11, m_label, self.font_btn, (255, 255, 255))

        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)


# =============================================================================
# MAIN APPLICATION LOOP
# =============================================================================

def main():
    # Pre-initialize low-latency stereo mixer for procedural heart sounds
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
    pygame.init()
    width, height = 1280, 720
    try:
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL, vsync=1)
    except TypeError:
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)

    pygame.display.set_caption("PyOpenGL 3D Beating Heart Simulation - Auscultation Sound, Translucent Vessels & Speed Control")

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glEnable(GL_NORMALIZE)
    glShadeModel(GL_SMOOTH)
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)

    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [4.5, 5.0, 6.5, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.95, 0.92, 0.90, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.75, 0.75, 0.75, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.24, 0.22, 0.26, 1.0])

    glEnable(GL_LIGHT1)
    glLightfv(GL_LIGHT1, GL_POSITION, [-5.5, 2.5, -4.5, 1.0])
    glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.42, 0.50, 0.68, 1.0])
    glLightfv(GL_LIGHT1, GL_SPECULAR, [0.30, 0.35, 0.45, 1.0])

    glEnable(GL_LIGHT2)
    glLightfv(GL_LIGHT2, GL_POSITION, [0.0, -6.0, 3.0, 1.0])
    glLightfv(GL_LIGHT2, GL_DIFFUSE, [0.28, 0.22, 0.25, 1.0])

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(42.0, width / float(height), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

    # Instantiate Anatomical, Physiological & Audio Systems
    cycle = CardiacCycleEngine(bpm=72.0)
    audio = CardiacAudioEngine(sample_rate=44100)
    heart = AnatomicalHeartModel()
    valves = HeartValveSystem()
    blood = BloodFlowSystem(num_particles=380)
    hud = HUDOverlay(width, height)

    init_cam_rot = [8.0, -14.0, 0.0]
    cam_rot = list(init_cam_rot)
    cam_zoom = 7.4
    sim_speed = 1.0
    paused = False
    dragging_slider = False

    def reset_simulation():
        nonlocal cam_rot, cam_zoom, sim_speed, paused
        cam_rot = list(init_cam_rot)
        cam_zoom = 7.4
        sim_speed = 1.0
        paused = False
        cycle.reset()
        blood.reset_particles()

    clock = pygame.time.Clock()
    fps_smooth = 60.0
    running = True

    while running:
        raw_dt = clock.tick(120) / 1000.0
        if raw_dt > 0:
            fps_smooth = 0.92 * fps_smooth + 0.08 * (1.0 / raw_dt)
        dt = min(raw_dt, 0.04)
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_p:
                    paused = not paused
                elif event.key == K_r:
                    reset_simulation()
                elif event.key == K_m:
                    audio.enabled = not audio.enabled
                elif event.key in (K_LEFTBRACKET, K_COMMA):
                    sim_speed = max(0.10, round(sim_speed - 0.10, 2))
                elif event.key in (K_RIGHTBRACKET, K_PERIOD):
                    sim_speed = min(3.00, round(sim_speed + 0.10, 2))
                elif event.key == K_1:
                    sim_speed = 0.25
                elif event.key == K_2:
                    sim_speed = 0.50
                elif event.key == K_3:
                    sim_speed = 1.00
                elif event.key == K_4:
                    sim_speed = 1.50
                elif event.key == K_5:
                    sim_speed = 2.00
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    if hud.btn_pause_rect.collidepoint(event.pos):
                        paused = not paused
                    elif hud.btn_reset_rect.collidepoint(event.pos):
                        reset_simulation()
                    elif hud.btn_slower_rect.collidepoint(event.pos):
                        sim_speed = max(0.10, round(sim_speed - 0.10, 2))
                    elif hud.btn_faster_rect.collidepoint(event.pos):
                        sim_speed = min(3.00, round(sim_speed + 0.10, 2))
                    elif hud.btn_mute_rect.collidepoint(event.pos):
                        audio.enabled = not audio.enabled
                    elif hud.slider_rect.collidepoint(event.pos):
                        dragging_slider = True
                        sim_speed = hud.speed_from_mouse_x(event.pos[0])
                elif event.button == 4:
                    cam_zoom = max(3.5, cam_zoom - 0.35)
                elif event.button == 5:
                    cam_zoom = min(14.0, cam_zoom + 0.35)
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_slider = False
            elif event.type == MOUSEMOTION:
                if dragging_slider:
                    sim_speed = hud.speed_from_mouse_x(event.pos[0])

        # Continuous Keyboard Direction Keys for X, Y, Z Camera Rotation
        keys = pygame.key.get_pressed()
        rot_speed = 80.0 * dt
        shift_held = keys[K_LSHIFT] or keys[K_RSHIFT]

        if keys[K_UP]:
            cam_rot[0] = (cam_rot[0] - rot_speed) % 360.0
        if keys[K_DOWN]:
            cam_rot[0] = (cam_rot[0] + rot_speed) % 360.0

        if shift_held:
            if keys[K_LEFT]:
                cam_rot[2] = (cam_rot[2] + rot_speed) % 360.0
            if keys[K_RIGHT]:
                cam_rot[2] = (cam_rot[2] - rot_speed) % 360.0
        else:
            if keys[K_LEFT]:
                cam_rot[1] = (cam_rot[1] - rot_speed) % 360.0
            if keys[K_RIGHT]:
                cam_rot[1] = (cam_rot[1] + rot_speed) % 360.0

        if keys[K_q] or keys[K_PAGEUP]:
            cam_rot[2] = (cam_rot[2] + rot_speed) % 360.0
        if keys[K_e] or keys[K_PAGEDOWN]:
            cam_rot[2] = (cam_rot[2] - rot_speed) % 360.0

        if keys[K_w] or keys[K_PLUS] or keys[K_EQUALS]:
            cam_zoom = max(3.5, cam_zoom - 3.8 * dt)
        if keys[K_s] or keys[K_MINUS]:
            cam_zoom = min(14.0, cam_zoom + 3.8 * dt)

        # Update Physiological Cardiac Cycle, Synchronized Heart Sounds & Blood Flow
        sim_dt = 0.0 if paused else (dt * sim_speed)
        cycle.update(sim_dt)
        audio.update_and_trigger(cycle.prev_phase, cycle.phase, dt, paused)
        blood.update(sim_dt, cycle)

        heart.solid_batch.update_geometry(cycle)
        heart.vessel_batch.update_geometry(cycle)
        heart.shell_batch.update_geometry(cycle)

        glClearColor(0.035, 0.045, 0.075, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        glTranslatef(0.0, -0.15, -cam_zoom)
        glRotatef(cam_rot[0], 1.0, 0.0, 0.0)
        glRotatef(cam_rot[1], 0.0, 1.0, 0.0)
        glRotatef(cam_rot[2], 0.0, 0.0, 1.0)

        # =====================================================================
        # PASS 1: OPAQUE SOLID INNER ANATOMY (Septum, Cradles, Valves, Blood)
        # =====================================================================
        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.40, 0.34, 0.34, 1.0])

        heart.solid_batch.draw(shininess=32.0)
        valves.render_valves_and_chordae(cycle, heart.papillary_anchors)
        blood.render(cycle)

        # =====================================================================
        # PASS 2: ALL-ANGLE TRANSLUCENT VEINS, ARTERIES & OUTER EPICARDIAL SHELL
        # =====================================================================
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        glEnable(GL_CULL_FACE)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.35, 0.32, 0.35, 1.0])

        for cull_mode in (GL_FRONT, GL_BACK):
            glCullFace(cull_mode)
            heart.vessel_batch.draw(shininess=48.0)
            heart.shell_batch.draw(shininess=60.0)

        glDisable(GL_CULL_FACE)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

        # =====================================================================
        # PASS 3: 2D HUD TELEMETRY, LIVE ECG/PCG & INTERACTIVE CONTROLS
        # =====================================================================
        hud.render(cycle, audio, cam_rot, sim_speed, paused, fps_smooth, mouse_pos)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()