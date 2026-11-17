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


def compute_model_space_camera_dir(cam_rot):
    """
    Computes the unit vector pointing from the origin toward the camera
    in 3D model space, given Euler angles cam_rot = [rx, ry, rz] in degrees.
    """
    rx = math.radians(cam_rot[0])
    ry = math.radians(cam_rot[1])
    rz = math.radians(cam_rot[2])

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    vx = -cx * sy * cz + sx * sz
    vy = cx * sy * sz + sx * cz
    vz = cx * cy
    v = np.array([vx, vy, vz], dtype=np.float32)
    return v / max(1e-8, np.linalg.norm(v))


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


def build_tube_mesh(centerline, radii, radial_segments=20, arc_range=(0.0, 2.0 * math.pi), scale_xz=(1.0, 1.0)):
    """
    Constructs a smooth tubular or chamber-shell 3D mesh along a 3D centerline
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

    sx, sz = scale_xz
    vertices = np.zeros((n_rings * n_cols, 3), dtype=np.float32)
    for i in range(n_rings):
        for j, theta in enumerate(angles):
            r = radii_arr[i]
            radial_vec = (sx * math.cos(theta)) * normals[i] + (sz * math.sin(theta)) * binormals[i]
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


def build_sphere_cluster_mesh(center, radius, sub_lobules=6):
    """Builds a grapelike alveolar acinar sac cluster at a terminal bronchiole tip."""
    c = np.asarray(center, dtype=np.float32)
    offsets = [
        [0.0, 0.0, 0.0],
        [0.55, 0.25, 0.20],
        [-0.50, 0.30, -0.20],
        [0.25, -0.55, 0.30],
        [-0.30, -0.45, -0.35],
        [0.15, 0.45, -0.48],
    ][:sub_lobules]

    n_lat, n_lon = 7, 10
    u = np.linspace(0.05, math.pi - 0.05, n_lat)
    v = np.linspace(0.0, 2.0 * math.pi, n_lon, endpoint=False)

    verts_all = []
    idx_all = []
    v_off = 0

    for off in offsets:
        lob_c = c + np.array(off, dtype=np.float32) * radius
        r_lob = radius * 0.68
        verts = np.zeros((n_lat * n_lon, 3), dtype=np.float32)
        for i, phi in enumerate(u):
            sp, cp = math.sin(phi), math.cos(phi)
            for j, th in enumerate(v):
                verts[i * n_lon + j] = lob_c + r_lob * np.array([sp * math.cos(th), cp, sp * math.sin(th)], dtype=np.float32)

        idx = []
        for i in range(n_lat - 1):
            for j in range(n_lon):
                jn = (j + 1) % n_lon
                a = v_off + i * n_lon + j
                b = v_off + (i + 1) * n_lon + j
                cc = v_off + (i + 1) * n_lon + jn
                d = v_off + i * n_lon + jn
                idx.extend([a, b, cc, a, cc, d])

        verts_all.append(verts)
        idx_all.extend(idx)
        v_off += len(verts)

    vertices = np.vstack(verts_all).astype(np.float32)
    indices = np.array(idx_all, dtype=np.uint32)
    normals = compute_vertex_normals(vertices, indices)
    return vertices, normals, indices


# =============================================================================
# PROCEDURAL CARDIOPULMONARY AUSCULTATION AUDIO ENGINE
# =============================================================================

class CardiacAudioEngine:
    """
    Synthesizes realistic physiological heart & lung sounds in memory:
      - S1 ("Lub"): Low-pitched M1/T1 thud when Mitral & Tricuspid valves close.
      - S2 ("Dub"): Higher-pitched A2/P2 snap when Aortic & Pulmonary valves close.
      - Systolic Flow Whoosh: Soft arterial blood rush during ventricular ejection.
      - Vesicular Breath Sounds: Soft inspiration & expiration pulmonary airflow sounds.
    """
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.enabled = True
        self.available = False
        self.active_sound_label = "VESICULAR BREATH / DIASTOLE"
        self.label_timer = 0.0

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=512)
            self.available = True
            self.s1_sound = self._synthesize_s1_lub()
            self.s2_sound = self._synthesize_s2_dub()
            self.flow_sound = self._synthesize_systolic_whoosh()
            self.inhale_sound = self._synthesize_breath_sound(duration=0.95, is_inhale=True)
            self.exhale_sound = self._synthesize_breath_sound(duration=1.10, is_inhale=False)
        except Exception as e:
            print(f"[Audio Warning] Could not initialize sound mixer: {e}")
            self.available = False

    def _to_stereo_sound(self, mono_wave, volume=0.85):
        peak = np.max(np.abs(mono_wave))
        if peak > 1e-6:
            mono_wave = (mono_wave / peak) * volume
        pcm16 = np.clip(mono_wave * 32767.0, -32767, 32767).astype(np.int16)
        stereo = np.column_stack((pcm16, pcm16))
        return pygame.sndarray.make_sound(stereo)

    def _synthesize_s1_lub(self):
        duration = 0.135
        t = np.linspace(0.0, duration, int(self.sample_rate * duration), endpoint=False)

        env1 = (t / 0.012) * np.exp(-t / 0.022)
        phase1 = 2.0 * np.pi * (68.0 * t - 110.0 * (t ** 2))
        m1 = env1 * (np.sin(phase1) + 0.45 * np.sin(1.85 * phase1))

        t2 = np.maximum(0.0, t - 0.022)
        env2 = (t2 / 0.012) * np.exp(-t2 / 0.024)
        phase2 = 2.0 * np.pi * (58.0 * t2 - 90.0 * (t2 ** 2))
        t1 = 0.78 * env2 * (np.sin(phase2) + 0.35 * np.sin(1.9 * phase2))

        rng = np.random.default_rng(101)
        noise = rng.normal(0.0, 1.0, len(t))
        kernel = np.hanning(45)
        kernel /= np.sum(kernel)
        thump = np.convolve(noise, kernel, mode="same") * env1 * 0.35

        wave = (m1 + t1 + thump) * (np.sin(np.linspace(0.0, np.pi, len(t))) ** 0.25)
        return self._to_stereo_sound(wave, volume=0.92)

    def _synthesize_s2_dub(self):
        duration = 0.105
        t = np.linspace(0.0, duration, int(self.sample_rate * duration), endpoint=False)

        env1 = (t / 0.008) * np.exp(-t / 0.016)
        phase1 = 2.0 * np.pi * (96.0 * t - 180.0 * (t ** 2))
        a2 = env1 * (np.sin(phase1) + 0.50 * np.sin(1.75 * phase1))

        t2 = np.maximum(0.0, t - 0.024)
        env2 = (t2 / 0.008) * np.exp(-t2 / 0.017)
        phase2 = 2.0 * np.pi * (82.0 * t2 - 150.0 * (t2 ** 2))
        p2 = 0.72 * env2 * (np.sin(phase2) + 0.40 * np.sin(1.8 * phase2))

        rng = np.random.default_rng(202)
        noise = rng.normal(0.0, 1.0, len(t))
        kernel = np.hanning(31)
        kernel /= np.sum(kernel)
        snap = np.convolve(noise, kernel, mode="same") * env1 * 0.28

        wave = (a2 + p2 + snap) * (np.sin(np.linspace(0.0, np.pi, len(t))) ** 0.25)
        return self._to_stereo_sound(wave, volume=0.82)

    def _synthesize_systolic_whoosh(self):
        duration = 0.19
        t = np.linspace(0.0, duration, int(self.sample_rate * duration), endpoint=False)
        rng = np.random.default_rng(303)
        white = rng.normal(0.0, 1.0, len(t))

        kernel = np.hanning(65)
        kernel /= np.sum(kernel)
        filtered = np.convolve(white, kernel, mode="same")

        env = np.sin((t / duration) * np.pi) ** 1.6
        pulse_mod = 0.65 + 0.35 * np.sin(2.0 * np.pi * 38.0 * t)
        wave = filtered * env * pulse_mod
        return self._to_stereo_sound(wave, volume=0.20)

    def _synthesize_breath_sound(self, duration=0.95, is_inhale=True):
        t = np.linspace(0.0, duration, int(self.sample_rate * duration), endpoint=False)
        rng = np.random.default_rng(404 if is_inhale else 505)
        white = rng.normal(0.0, 1.0, len(t))

        k_size = 75 if is_inhale else 105
        kernel = np.hanning(k_size)
        kernel /= np.sum(kernel)
        breath = np.convolve(white, kernel, mode="same")

        env = np.sin((t / duration) * np.pi) ** (1.3 if is_inhale else 1.8)
        vol = 0.14 if is_inhale else 0.10
        return self._to_stereo_sound(breath * env, volume=vol)

    def update_and_trigger(self, prev_c_phase, curr_c_phase, prev_r_phase, curr_r_phase, dt, paused):
        if self.label_timer > 0.0:
            self.label_timer = max(0.0, self.label_timer - dt)
            if self.label_timer == 0.0:
                self.active_sound_label = "VESICULAR AIRFLOW"

        if paused or not self.enabled or not self.available or dt <= 0.0:
            return

        def crossed(prev_p, curr_p, target):
            if prev_p <= curr_p:
                return prev_p < target <= curr_p
            return prev_p < target or curr_p >= target

        if crossed(prev_c_phase, curr_c_phase, 0.14):
            self.s1_sound.play()
            self.active_sound_label = "S1: 'LUB' (AV Valves Close)"
            self.label_timer = 0.18

        if crossed(prev_c_phase, curr_c_phase, 0.22):
            self.flow_sound.play()

        if crossed(prev_c_phase, curr_c_phase, 0.46):
            self.s2_sound.play()
            self.active_sound_label = "S2: 'DUB' (Semilunar Close)"
            self.label_timer = 0.16

        if crossed(prev_r_phase, curr_r_phase, 0.03):
            self.inhale_sound.play()
        if crossed(prev_r_phase, curr_r_phase, 0.44):
            self.exhale_sound.play()


# =============================================================================
# SYNCHRONIZED CARDIOPULMONARY CYCLE ENGINE (HEART + LUNGS)
# =============================================================================

class CardiopulmonaryEngine:
    """
    Synchronizes the human Cardiac Cycle (72 BPM) and Respiratory Cycle (18 Breaths/min).
    """
    def __init__(self, bpm=72.0, resp_rate=18.0):
        self.bpm = bpm
        self.resp_rate = resp_rate
        self.cardiac_period = 60.0 / bpm
        self.resp_period = 60.0 / resp_rate

        self.time = 0.0
        self.prev_phase = 0.0
        self.phase = 0.0
        self.prev_resp_phase = 0.0
        self.resp_phase = 0.0

        self.atrial_contraction = 0.0
        self.ventricular_contraction = 0.0
        self.arterial_pulse = 0.0
        self.apical_twist = 0.0

        self.tricuspid_open = 1.0
        self.mitral_open = 1.0
        self.pulmonary_open = 0.0
        self.aortic_open = 0.0

        self.lung_inflation = 0.0
        self.airflow_velocity = 0.0
        self.alveolar_oxygen = 0.5

        self.phase_name = "Diastole (Passive Filling)"
        self.resp_phase_name = "Inspiration (Alveolar Filling)"

        self.ecg_history = [0.0] * 180
        self.pcg_history = [0.0] * 180
        self.resp_history = [0.0] * 180
        self.trace_accum = 0.0

    def reset(self):
        self.time = 0.0
        self.prev_phase = 0.0
        self.phase = 0.0
        self.prev_resp_phase = 0.0
        self.resp_phase = 0.0
        self.trace_accum = 0.0
        self.ecg_history = [0.0] * 180
        self.pcg_history = [0.0] * 180
        self.resp_history = [0.0] * 180
        self.update(0.0)

    def _compute_ecg(self, p):
        p_wave = 0.18 * math.exp(-((p - 0.06) / 0.032) ** 2)
        q_wave = -0.15 * math.exp(-((p - 0.155) / 0.010) ** 2)
        r_wave = 1.05 * math.exp(-((p - 0.175) / 0.012) ** 2)
        s_wave = -0.28 * math.exp(-((p - 0.195) / 0.011) ** 2)
        t_wave = 0.30 * math.exp(-((p - 0.41) / 0.048) ** 2)
        return p_wave + q_wave + r_wave + s_wave + t_wave

    def _compute_pcg(self, p):
        s1_env = math.exp(-((p - 0.16) / 0.026) ** 2)
        s2_env = 0.82 * math.exp(-((p - 0.475) / 0.020) ** 2)
        return s1_env * math.sin(2.0 * math.pi * 42.0 * p) + s2_env * math.sin(2.0 * math.pi * 64.0 * p)

    def update(self, dt):
        self.prev_phase = self.phase
        self.prev_resp_phase = self.resp_phase
        self.time += dt

        self.resp_phase = (self.time % self.resp_period) / self.resp_period
        rp = self.resp_phase

        if rp < 0.44:
            u = rp / 0.44
            self.lung_inflation = 0.5 * (1.0 - math.cos(u * math.pi))
            self.airflow_velocity = math.sin(u * math.pi)
            self.resp_phase_name = "Inspiration (Diaphragm Descends, O2 In)"
        elif rp < 0.48:
            self.lung_inflation = 1.0
            self.airflow_velocity = 0.0
            self.resp_phase_name = "End-Inspiratory Pause (Peak Gas Exchange)"
        else:
            u = (rp - 0.48) / 0.52
            self.lung_inflation = 0.5 * (1.0 + math.cos(u * math.pi))
            self.airflow_velocity = -0.85 * math.sin(u * math.pi)
            self.resp_phase_name = "Expiration (Elastic Recoil, CO2 Out)"

        self.alveolar_oxygen = 0.35 + 0.65 * self.lung_inflation

        self.phase = (self.time % self.cardiac_period) / self.cardiac_period
        p = self.phase

        if p < 0.15:
            self.atrial_contraction = math.sin((p / 0.15) * math.pi) ** 1.4
        else:
            self.atrial_contraction = -0.07 * math.sin(((p - 0.15) / 0.85) * math.pi)

        if 0.14 <= p < 0.52:
            norm_v = (p - 0.14) / 0.38
            self.ventricular_contraction = math.sin(norm_v * math.pi) ** 1.25
        else:
            if p < 0.14:
                self.ventricular_contraction = -0.06 * math.sin((p / 0.14) * math.pi)
            else:
                self.ventricular_contraction = 0.0

        if 0.19 <= p < 0.64:
            norm_a = (p - 0.19) / 0.45
            main_wave = math.sin(norm_a * math.pi) ** 1.4
            dicrotic = 0.16 * math.exp(-((p - 0.52) / 0.038) ** 2)
            self.arterial_pulse = min(1.0, main_wave + dicrotic)
        else:
            self.arterial_pulse = 0.0

        self.apical_twist = 0.18 * self.ventricular_contraction

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

        sl_open = cosine_window(p, 0.185, 0.23, 0.44, 0.49)
        self.pulmonary_open = float(np.clip(sl_open, 0.0, 1.0))
        self.aortic_open = float(np.clip(sl_open, 0.0, 1.0))

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

        if dt > 0.0:
            self.trace_accum += dt
            step = 1.0 / 60.0
            while self.trace_accum >= step:
                self.trace_accum -= step
                self.ecg_history.append(self._compute_ecg(p))
                self.pcg_history.append(self._compute_pcg(p))
                self.resp_history.append(self.lung_inflation)
                if len(self.ecg_history) > 180:
                    self.ecg_history.pop(0)
                if len(self.pcg_history) > 180:
                    self.pcg_history.pop(0)
                if len(self.resp_history) > 180:
                    self.resp_history.pop(0)


# =============================================================================
# SYNCHRONIZED CARDIOPULMONARY DEFORMATION & FRESNEL X-RAY SHADING ENGINE
# =============================================================================

def deform_points_direct(pts, cycle: CardiopulmonaryEngine):
    """
    Applies identical, anatomically matched biomechanics to landmark points
    (valves, papillary tips, blood cells, and airway particles) as FastDeformableBatch:
      - Ventricular Cone (y < 0.24): radial contraction toward the tilted interventricular
        axis, apical longitudinal shortening, and counter-clockwise apical twist.
      - Atrial Chambers (0.24 <= y < 1.18): localized contraction around the true anatomical
        centers of the Right Atrium (-0.82, 0.68, 0.04) and Left Atrium (+0.72, 0.72, -0.24).
      - Great Vessels (y >= 0.90): pulsatile systolic radial dilation.
      - Lungs (|x| >= 1.25): lateral, anterior, and diaphragmatic expansion.
    """
    v = pts.copy()
    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    cardiac_zone = smoothstep01((1.40 - np.abs(x)) / 0.42)
    lung_zone = 1.0 - cardiac_zone

    # --- A. Ventricular Contraction (below the AV valve plane y = 0.24) ---
    v_mask = smoothstep01((0.24 - y) / 1.75) * cardiac_zone
    vc = cycle.ventricular_contraction

    apex_t = np.clip((0.24 - y) / 1.95, 0.0, 1.0)
    vent_cx = 0.56 * (apex_t ** 1.35)
    vent_cz = 0.24 * (apex_t ** 1.35)

    dx_v = x - vent_cx
    dz_v = z - vent_cz

    squeeze = 1.0 - 0.22 * vc * v_mask
    x_c = vent_cx + dx_v * squeeze
    z_c = vent_cz + dz_v * squeeze
    y_c = y + 0.18 * vc * (v_mask ** 1.35)

    twist_angle = cycle.apical_twist * (v_mask ** 1.25)
    cos_t = np.cos(twist_angle)
    sin_t = np.sin(twist_angle)
    rx = x_c - vent_cx
    rz = z_c - vent_cz
    x_c = vent_cx + rx * cos_t - rz * sin_t
    z_c = vent_cz + rx * sin_t + rz * cos_t

    # --- B. Atrial Contraction (centered on true RA and LA chambers) ---
    a_mask = np.exp(-((y - 0.70) / 0.36) ** 2) * cardiac_zone
    ac = cycle.atrial_contraction
    ra_dist = np.exp(-((x + 0.82) ** 2 + (z - 0.04) ** 2) / 0.65)
    la_dist = np.exp(-((x - 0.72) ** 2 + (z + 0.24) ** 2) / 0.65)

    pull_ra = 0.22 * ac * a_mask * ra_dist
    pull_la = 0.22 * ac * a_mask * la_dist

    x_c = x_c - pull_ra * (x_c + 0.82) - pull_la * (x_c - 0.72)
    y_c = y_c - 0.11 * ac * a_mask * (ra_dist + la_dist) * (y_c - 0.60)
    z_c = z_c - pull_ra * (z_c - 0.04) - pull_la * (z_c + 0.24)

    # --- C. Great Arterial Pulse Wave (for superior vessels y > 0.90) ---
    art_mask = smoothstep01((y - 0.90) / 0.95) * cardiac_zone
    ap = cycle.arterial_pulse
    scale_art = 1.0 + 0.055 * ap * art_mask
    x_c = x_c * scale_art
    z_c = z_c * scale_art

    # --- D. Pulmonary Respiratory Expansion (for lateral lung regions) ---
    infl = cycle.lung_inflation
    sign_x = np.where(x >= 0.0, 1.0, -1.0)
    lung_hilum_x = sign_x * 1.15
    inf_weight = smoothstep01((1.65 - y) / 3.4) * lung_zone
    sup_weight = smoothstep01((y - 0.50) / 1.8) * lung_zone

    x_l = x_c + (x - lung_hilum_x) * (0.14 * infl * lung_zone)
    y_l = y_c - 0.26 * infl * (inf_weight ** 1.2) + 0.05 * infl * sup_weight
    z_l = z_c + z * (0.16 * infl * lung_zone) + 0.04 * infl * lung_zone

    v[:, 0] = x_l
    v[:, 1] = y_l
    v[:, 2] = z_l
    return v


class FastDeformableBatch:
    """
    Precomputes spatial deformation weights and applies a Fresnel Rim-Lit X-Ray
    optical shader:
      - Surfaces facing directly toward/away from the camera (center of chambers)
        open a clear low-alpha window (alpha ~ 0.10 - 0.15) into the valves & blood.
      - Surfaces at grazing angles (3D outer silhouette, chamber walls, myocardial
        thickness, sulci) ramp up to rich opacity (alpha ~ 0.65 - 0.80) with a warm
        rim highlight so the heart's shape, atria, and ventricles are always bold!
    """
    def __init__(self, mesh_list, is_translucent=False):
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
        self.base_colors = np.vstack(cols_all).astype(np.float32)
        self.base_rgb = self.base_colors[:, :3].copy()
        self.base_alpha = self.base_colors[:, 3].copy()
        self.indices = np.concatenate(indices_all).astype(np.uint32)
        self.is_translucent = is_translucent

        self.colors_back_pass = self.base_colors.copy()
        self.colors_front_pass = self.base_colors.copy()

        x = self.base_verts[:, 0]
        y = self.base_verts[:, 1]
        z = self.base_verts[:, 2]

        self.cardiac_zone = smoothstep01((1.40 - np.abs(x)) / 0.42).astype(np.float32)
        self.lung_zone = (1.0 - self.cardiac_zone).astype(np.float32)

        self.v_mask = (smoothstep01((0.24 - y) / 1.75) * self.cardiac_zone).astype(np.float32)
        self.v_squeeze_factor = (0.22 * self.v_mask).astype(np.float32)
        self.v_lift_factor = (0.18 * (self.v_mask ** 1.35)).astype(np.float32)
        self.v_twist_factor = (self.v_mask ** 1.25).astype(np.float32)

        apex_t = np.clip((0.24 - y) / 1.95, 0.0, 1.0)
        self.vent_cx = (0.56 * (apex_t ** 1.35)).astype(np.float32)
        self.vent_cz = (0.24 * (apex_t ** 1.35)).astype(np.float32)
        self.dx_v = (x - self.vent_cx).astype(np.float32)
        self.dz_v = (z - self.vent_cz).astype(np.float32)

        a_mask = np.exp(-((y - 0.70) / 0.36) ** 2) * self.cardiac_zone
        ra_dist = np.exp(-((x + 0.82) ** 2 + (z - 0.04) ** 2) / 0.65)
        la_dist = np.exp(-((x - 0.72) ** 2 + (z + 0.24) ** 2) / 0.65)

        self.ra_weight = (0.22 * a_mask * ra_dist).astype(np.float32)
        self.la_weight = (0.22 * a_mask * la_dist).astype(np.float32)
        self.ay_weight = (0.11 * a_mask * (ra_dist + la_dist)).astype(np.float32)
        self.art_weight = (0.055 * smoothstep01((y - 0.90) / 0.95) * self.cardiac_zone).astype(np.float32)

        sign_x = np.where(x >= 0.0, 1.0, -1.0).astype(np.float32)
        lung_hilum_x = sign_x * 1.15
        inf_weight = smoothstep01((1.65 - y) / 3.4) * self.lung_zone
        sup_weight = smoothstep01((y - 0.50) / 1.8) * self.lung_zone

        self.lung_dx = ((x - lung_hilum_x) * 0.14 * self.lung_zone).astype(np.float32)
        self.lung_dy = (-0.26 * (inf_weight ** 1.2) + 0.05 * sup_weight).astype(np.float32)
        self.lung_dz = ((z * 0.16 + 0.04) * self.lung_zone).astype(np.float32)

        self.cur_verts = np.empty_like(self.base_verts)
        self.cur_norms = np.empty_like(self.base_norms)

    def update_geometry(self, cycle: CardiopulmonaryEngine, cam_view_dir=None):
        vc = np.float32(cycle.ventricular_contraction)
        ac = np.float32(cycle.atrial_contraction)
        ap = np.float32(cycle.arterial_pulse)
        tw = np.float32(cycle.apical_twist)
        infl = np.float32(cycle.lung_inflation)

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

        x_new = x_new - pull_ra * (x_new + 0.82) - pull_la * (x_new - 0.72)
        y_new = y_new - ac * self.ay_weight * (y_new - 0.60)
        z_new = z_new - pull_ra * (z_new - 0.04) - pull_la * (z_new + 0.24)

        art_scale = 1.0 + ap * self.art_weight
        x_new = x_new * art_scale + infl * self.lung_dx
        y_new = y_new + infl * self.lung_dy
        z_new = z_new * art_scale + infl * self.lung_dz

        self.cur_verts[:, 0] = x_new
        self.cur_verts[:, 1] = y_new
        self.cur_verts[:, 2] = z_new

        nx = self.base_norms[:, 0]
        ny = self.base_norms[:, 1]
        nz = self.base_norms[:, 2]
        self.cur_norms[:, 0] = nx * cos_t - nz * sin_t
        self.cur_norms[:, 1] = ny
        self.cur_norms[:, 2] = nx * sin_t + nz * cos_t

        # Fresnel Rim-Lit X-Ray Shader:
        # Center of camera-facing walls = clear window; Edges & Chamber Contours = rich & solid-looking!
        if self.is_translucent and cam_view_dir is not None:
            abs_dot = np.abs(
                self.cur_norms[:, 0] * cam_view_dir[0]
                + self.cur_norms[:, 1] * cam_view_dir[1]
                + self.cur_norms[:, 2] * cam_view_dir[2]
            )
            rim = np.clip(1.0 - abs_dot, 0.0, 1.0)
            rim_curve = rim ** 1.45

            # Boost RGB slightly on silhouette rims so chambers & sulci pop in 3D
            rim_boost = (0.18 * rim_curve)[:, None]
            boosted_rgb = np.clip(self.base_rgb + rim_boost, 0.0, 1.0)
            self.colors_back_pass[:, :3] = boosted_rgb
            self.colors_front_pass[:, :3] = boosted_rgb

            # Backside pass: visible anatomical cradle (0.24 at center -> 0.72 at rim)
            self.colors_back_pass[:, 3] = np.clip(
                self.base_alpha * (0.78 + 1.55 * rim_curve), 0.14, 0.78
            )
            # Frontside pass: clear X-ray window at center (0.09 at center -> 0.65 at rim)
            self.colors_front_pass[:, 3] = np.clip(
                self.base_alpha * (0.30 + 1.45 * rim_curve), 0.07, 0.68
            )

    def draw_opaque(self, shininess=36.0):
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, self.cur_verts)
        glNormalPointer(GL_FLOAT, 0, self.cur_norms)
        glColorPointer(4, GL_FLOAT, 0, self.base_colors)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, self.indices)

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    def draw_translucent_pass(self, is_front_pass=False, shininess=52.0):
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        col_buf = self.colors_front_pass if is_front_pass else self.colors_back_pass
        glVertexPointer(3, GL_FLOAT, 0, self.cur_verts)
        glNormalPointer(GL_FLOAT, 0, self.cur_norms)
        glColorPointer(4, GL_FLOAT, 0, col_buf)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, self.indices)

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)


# =============================================================================
# PROCEDURAL 3D ANATOMICAL HEART & LUNGS BUILDER
# =============================================================================

class CardiopulmonaryAnatomyModel:
    """
    Constructs a 3-Layer Anatomical Heart + Lungs System:
      1. SOLID OPAQUE CORE SKELETON (Crux Cordis):
         - Muscular Interventricular Septum (dividing RV & LV)
         - Interatrial Septum (dividing RA & LA)
         - Horizontal Atrioventricular Fibrous Skeleton Ring (dividing Atria & Ventricles)
         - Septal Trabeculae Carneae, Moderator Band & 5 Conical Papillary Muscles
         - Epicardial Coronary Artery & Cardiac Vein Network (framing the outer surface)
      2. 4 DISTINCT TRANSLUCENT ENDOCARDIAL CHAMBERS (Fresnel X-Ray Shaded):
         - Right Atrium (RA), Left Atrium (LA), Right Ventricle (RV), Left Ventricle (LV)
      3. TRANSLUCENT OUTER EPICARDIAL SHELL & LUNGS (Fresnel X-Ray Shaded):
         - Shows true myocardial wall thickness around the 4 chambers while keeping
           the center facing your camera open like a window!
    """
    def __init__(self):
        self.papillary_anchors = []
        self.alveolar_centers = []
        self._build_all_geometry()

    def _build_outer_epicardial_shell(self):
        """
        Builds the outer epicardial/myocardial surface of the conical heart.
        """
        n_lat = 56
        n_lon = 68
        u = np.linspace(0.02, math.pi - 0.02, n_lat)
        v = np.linspace(0.0, 2.0 * math.pi, n_lon, endpoint=False)

        vertices = np.zeros((n_lat * n_lon, 3), dtype=np.float32)
        for i, phi in enumerate(u):
            cos_p = math.cos(phi)
            sin_p = math.sin(phi)

            if cos_p >= 0.35:
                t_top = (cos_p - 0.35) / 0.65
                r_env = 1.26 * math.sqrt(max(0.02, 1.0 - (t_top ** 2.2)))
            else:
                t_bot = (0.35 - cos_p) / 1.35
                r_env = 1.26 * ((1.0 - (t_bot ** 1.45)) ** 0.62)

            for j, theta in enumerate(v):
                iv_groove_ant = -0.12 * math.exp(-((theta - 1.35) / 0.28) ** 2) * (sin_p ** 1.3)
                iv_groove_post = -0.10 * math.exp(-((theta - 4.55) / 0.30) ** 2) * (sin_p ** 1.3)
                av_sulcus = -0.16 * math.exp(-((cos_p - 0.35) / 0.12) ** 2)

                ra_auricle = 0.25 * math.exp(-((cos_p - 0.64) / 0.22) ** 2) * math.exp(-((theta - 2.95) / 0.55) ** 2)
                la_auricle = 0.23 * math.exp(-((cos_p - 0.66) / 0.22) ** 2) * math.exp(-((theta - 5.95) / 0.55) ** 2)
                lv_curve = 0.18 * math.exp(-((cos_p + 0.10) / 0.48) ** 2) * math.exp(-((theta - 0.25) / 0.75) ** 2)

                r = max(0.04, r_env + iv_groove_ant + iv_groove_post + av_sulcus + ra_auricle + la_auricle + lv_curve)

                x = r * math.cos(theta) * 1.08
                z = r * math.sin(theta) * 0.84
                y = 1.36 * cos_p - 0.24

                apex_t = max(0.0, (0.35 - cos_p) / 1.35) ** 1.35
                x += 0.56 * apex_t
                z += 0.24 * apex_t

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
        # Base alpha 0.30 -> Fresnel shader makes center ~0.09 (clear window) & edges ~0.66 (bold 3D contour!)
        return (vertices, normals, indices, (0.92, 0.20, 0.26, 0.30))

    def _build_four_endocardial_chambers(self):
        """
        Builds 4 distinct, anatomically sculpted 3D endocardial chamber capsules
        (Right Atrium, Left Atrium, Right Ventricle, Left Ventricle) inside the
        outer heart shell so the viewer clearly sees all 4 chambers and the
        surrounding myocardial wall thickness!
        """
        chambers = []

        # 1. RIGHT ATRIUM (RA) Endocardial Chamber (Superior Right, receives SVC & IVC)
        ra_pts = catmull_rom_spline([
            [-0.95, 1.02, -0.02],
            [-0.86, 0.72,  0.04],
            [-0.76, 0.44,  0.10],
            [-0.64, 0.24,  0.14],
        ], num_samples=22)
        ra_rad = np.interp(np.linspace(0, 1, 22), [0.0, 0.45, 0.85, 1.0], [0.24, 0.46, 0.40, 0.28])
        v_ra, n_ra, i_ra = build_tube_mesh(ra_pts, ra_rad, radial_segments=22, scale_xz=(0.95, 1.05))
        chambers.append((v_ra, n_ra, i_ra, (0.58, 0.22, 0.44, 0.32)))

        # 2. LEFT ATRIUM (LA) Endocardial Chamber (Superior Left-Posterior, receives 4 PVs)
        la_pts = catmull_rom_spline([
            [ 0.78, 1.00, -0.28],
            [ 0.74, 0.74, -0.22],
            [ 0.68, 0.46, -0.16],
            [ 0.62, 0.26, -0.10],
        ], num_samples=22)
        la_rad = np.interp(np.linspace(0, 1, 22), [0.0, 0.45, 0.85, 1.0], [0.22, 0.44, 0.38, 0.27])
        v_la, n_la, i_la = build_tube_mesh(la_pts, la_rad, radial_segments=22, scale_xz=(1.02, 0.92))
        chambers.append((v_la, n_la, i_la, (0.88, 0.24, 0.32, 0.32)))

        # 3. RIGHT VENTRICLE (RV) Endocardial Chamber (Inflow from Tricuspid -> RV Belly -> Infundibulum to Pulmonary Valve)
        rv_pts = catmull_rom_spline([
            [-0.64,  0.22,  0.14],  # Tricuspid orifice
            [-0.72, -0.35,  0.20],  # RV inflow belly
            [-0.52, -0.96,  0.22],  # RV apical floor
            [-0.22, -1.18,  0.24],  # RV lower turn
            [-0.18, -0.42,  0.34],  # RV outflow tract (conus arteriosus)
            [-0.22,  0.74,  0.42],  # Pulmonary valve orifice
        ], num_samples=36)
        rv_rad = np.interp(
            np.linspace(0, 1, 36),
            [0.0, 0.25, 0.55, 0.75, 1.0],
            [0.29, 0.44, 0.36, 0.30, 0.23]
        )
        v_rv, n_rv, i_rv = build_tube_mesh(rv_pts, rv_rad, radial_segments=22, scale_xz=(0.88, 1.08))
        chambers.append((v_rv, n_rv, i_rv, (0.66, 0.18, 0.28, 0.34)))

        # 4. LEFT VENTRICLE (LV) Endocardial Chamber (Inflow from Mitral -> Thick Conical LV Apex -> Aortic Outflow)
        lv_pts = catmull_rom_spline([
            [ 0.62,  0.24, -0.10],  # Mitral orifice
            [ 0.72, -0.38,  0.02],  # LV mid-cavity
            [ 0.66, -1.05,  0.14],  # LV lower body
            [ 0.54, -1.48,  0.22],  # LV luminal apex (inside the thicker outer epicardial apex!)
            [ 0.34, -0.58,  0.10],  # LV outflow tract
            [ 0.18,  0.52,  0.02],  # Aortic valve orifice
        ], num_samples=36)
        lv_rad = np.interp(
            np.linspace(0, 1, 36),
            [0.0, 0.25, 0.55, 0.75, 1.0],
            [0.28, 0.45, 0.34, 0.28, 0.24]
        )
        v_lv, n_lv, i_lv = build_tube_mesh(lv_pts, lv_rad, radial_segments=22, scale_xz=(0.95, 0.95))
        chambers.append((v_lv, n_lv, i_lv, (0.86, 0.15, 0.20, 0.34)))

        return chambers

    def _build_solid_core_skeleton(self):
        """
        Builds the 100% OPAQUE SOLID INTERNAL CORE of the heart:
          1. Muscular Interventricular Septum (thick blade dividing RV and LV)
          2. Interatrial Septum (dividing RA and LA)
          3. Horizontal Atrioventricular Fibrous Skeleton / Crux Cordis (dividing Atria above from Ventricles below)
          4. Septal Trabeculae Carneae & Moderator Band (muscular beams inside the ventricles)
          5. 5 Conical Papillary Muscles
        """
        core_meshes = []

        # 1. Muscular Interventricular Septum (y from +0.28 down to -1.54)
        iv_septum_pts = catmull_rom_spline([
            [ 0.00,  0.28, -0.02],
            [ 0.06, -0.25,  0.06],
            [ 0.22, -0.88,  0.14],
            [ 0.48, -1.52,  0.22],
        ], num_samples=28)
        iv_rad = np.interp(np.linspace(0, 1, 28), [0.0, 0.3, 0.7, 1.0], [0.34, 0.42, 0.36, 0.16])
        v_ivs, n_ivs, i_ivs = build_tube_mesh(iv_septum_pts, iv_rad, radial_segments=24, scale_xz=(0.42, 1.45))
        core_meshes.append((v_ivs, n_ivs, i_ivs, (0.72, 0.15, 0.19, 1.0)))

        # 2. Interatrial Septum (y from +0.24 up to +0.98 between RA and LA)
        ia_septum_pts = catmull_rom_spline([
            [-0.04, 0.98, -0.14],
            [-0.02, 0.62, -0.08],
            [ 0.00, 0.24, -0.02],
        ], num_samples=18)
        ia_rad = np.interp(np.linspace(0, 1, 18), [0.0, 0.5, 1.0], [0.18, 0.30, 0.28])
        v_ias, n_ias, i_ias = build_tube_mesh(ia_septum_pts, ia_rad, radial_segments=20, scale_xz=(0.35, 1.35))
        core_meshes.append((v_ias, n_ias, i_ias, (0.78, 0.24, 0.28, 1.0)))

        # 3. Horizontal Atrioventricular Fibrous Skeleton Bridge (Crux Cordis at y = 0.24)
        av_bridge_pts = catmull_rom_spline([
            [-0.95, 0.24,  0.10],
            [-0.32, 0.24,  0.04],
            [ 0.00, 0.25, -0.02],
            [ 0.32, 0.25, -0.06],
            [ 0.92, 0.26, -0.08],
        ], num_samples=22)
        av_rad = np.interp(np.linspace(0, 1, 22), [0.0, 0.5, 1.0], [0.07, 0.13, 0.07])
        v_avb, n_avb, i_avb = build_tube_mesh(av_bridge_pts, av_rad, radial_segments=14, scale_xz=(1.0, 1.8))
        core_meshes.append((v_avb, n_avb, i_avb, (0.86, 0.76, 0.68, 1.0)))

        # 4. Septal Trabeculae Carneae & Right Ventricular Moderator Band (Septomarginal Trabecula)
        trabeculae_specs = [
            # Moderator band (bridges lower interventricular septum to anterior RV papillary base)
            ([[ 0.14, -0.68,  0.12], [-0.28, -0.82,  0.22], [-0.68, -0.82,  0.26]], 0.055),
            # RV & LV Septal muscular ridges
            ([[ 0.02, -0.15,  0.22], [-0.15, -0.65,  0.28], [-0.35, -1.12,  0.24]], 0.052),
            ([[ 0.02, -0.18, -0.16], [-0.22, -0.72, -0.12], [-0.45, -1.02,  0.05]], 0.050),
            ([[ 0.12, -0.15,  0.18], [ 0.35, -0.75,  0.22], [ 0.52, -1.38,  0.22]], 0.058),
            ([[ 0.12, -0.20, -0.18], [ 0.38, -0.82, -0.14], [ 0.52, -1.35,  0.08]], 0.055),
        ]
        for pts, rad in trabeculae_specs:
            curve = catmull_rom_spline(pts, num_samples=14)
            vt, nt, it = build_tube_mesh(curve, rad, radial_segments=10)
            core_meshes.append((vt, nt, it, (0.75, 0.16, 0.21, 1.0)))

        # 5. Five Solid Conical Papillary Muscles
        pap_specs = [
            {"base": np.array([-0.68, -0.82,  0.26]), "tip": np.array([-0.64, -0.24,  0.20]), "valve": "tricuspid", "leaf_idx": 0},
            {"base": np.array([-0.52, -0.96, -0.08]), "tip": np.array([-0.58, -0.26,  0.00]), "valve": "tricuspid", "leaf_idx": 1},
            {"base": np.array([-0.14, -0.85,  0.14]), "tip": np.array([-0.38, -0.22,  0.14]), "valve": "tricuspid", "leaf_idx": 2},
            {"base": np.array([ 0.82, -0.95,  0.10]), "tip": np.array([ 0.68, -0.24, -0.02]), "valve": "mitral", "leaf_idx": 0},
            {"base": np.array([ 0.52, -1.04, -0.22]), "tip": np.array([ 0.56, -0.26, -0.20]), "valve": "mitral", "leaf_idx": 1},
        ]
        self.papillary_anchors = []
        for sp in pap_specs:
            curve = catmull_rom_spline([
                sp["base"],
                0.5 * (sp["base"] + sp["tip"]) + np.array([0.0, 0.03, 0.02]),
                sp["tip"]
            ], num_samples=12)
            radii = np.linspace(0.115, 0.042, 12)
            vp, np_norm, ip = build_tube_mesh(curve, radii, radial_segments=12)
            core_meshes.append((vp, np_norm, ip, (0.78, 0.18, 0.23, 1.0)))
            self.papillary_anchors.append(sp)

        # 6. Solid Epicardial Coronary Arteries & Cardiac Veins (gives crisp 3D surface landmarks!)
        cor_artery_col = (0.92, 0.14, 0.18, 1.0)
        cor_vein_col = (0.18, 0.42, 0.92, 1.0)
        cor_specs = [
            # LAD (Left Anterior Descending) in the Anterior Interventricular Sulcus
            ([[ 0.22, 0.56, 0.14], [ 0.10, 0.28, 0.62], [ 0.16, -0.32, 0.74], [ 0.34, -0.98, 0.62], [ 0.52, -1.55, 0.38]], 0.040, 0.016, cor_artery_col),
            # Diagonal branch across anterior LV wall
            ([[ 0.16, -0.25, 0.73], [ 0.56, -0.58, 0.64], [ 0.82, -1.02, 0.42]], 0.030, 0.014, cor_artery_col),
            # Left Circumflex (LCX) in the left Coronary Sulcus
            ([[ 0.22, 0.56, 0.14], [ 0.78, 0.32, 0.26], [ 1.12, 0.14, -0.14], [ 0.92, -0.35, -0.48]], 0.038, 0.016, cor_artery_col),
            # Right Coronary Artery (RCA) in the right Coronary Sulcus
            ([[ 0.08, 0.54, 0.18], [-0.48, 0.28, 0.54], [-0.98, -0.05, 0.32], [-0.90, -0.50, -0.22], [-0.38, -0.90, -0.48]], 0.038, 0.016, cor_artery_col),
            # Great Cardiac Vein
            ([[ 0.44, -1.48, 0.42], [ 0.22, -0.88, 0.66], [ 0.06, -0.22, 0.75], [ 0.65, 0.26, 0.35], [ 1.05, 0.18, -0.20]], 0.016, 0.035, cor_vein_col),
        ]
        for pts, r0, r1, col in cor_specs:
            cc = catmull_rom_spline(pts, num_samples=24)
            cr = np.linspace(r0, r1, 24)
            vc, nc, ic = build_tube_mesh(cc, cr, radial_segments=10)
            core_meshes.append((vc, nc, ic, col))

        return core_meshes

    def _build_lung_shell(self, is_right_lung=True):
        """
        Builds anatomical Right Lung (3 lobes) and Left Lung (2 lobes + Cardiac Notch).
        """
        n_lat = 54
        n_lon = 64
        u = np.linspace(0.02, math.pi - 0.02, n_lat)
        v = np.linspace(0.0, 2.0 * math.pi, n_lon, endpoint=False)

        sign_x = -1.0 if is_right_lung else 1.0
        base_cx = -2.28 if is_right_lung else 2.25

        vertices = np.zeros((n_lat * n_lon, 3), dtype=np.float32)
        for i, phi in enumerate(u):
            cos_p = math.cos(phi)
            sin_p = math.sin(phi)
            y = 1.95 * cos_p + 0.12

            r_profile = sin_p * (1.02 - 0.18 * cos_p + 0.22 * (1.0 - cos_p) ** 1.4)

            for j, theta in enumerate(v):
                lx = r_profile * math.cos(theta) * (1.12 if is_right_lung else 1.04)
                lz = r_profile * math.sin(theta) * 1.18

                medial_dir = -sign_x * math.cos(theta)
                if medial_dir > 0.0:
                    lx *= (1.0 - 0.34 * (medial_dir ** 1.3))

                fiss_y_oblique = -0.25 - 0.55 * math.sin(theta)
                oblique_groove = 1.0 - 0.16 * math.exp(-((y - fiss_y_oblique) / 0.13) ** 2)

                horiz_groove = 1.0
                if is_right_lung and math.sin(theta) > -0.25:
                    horiz_groove = 1.0 - 0.14 * math.exp(-((y - 0.32) / 0.11) ** 2)

                fissure_scale = min(oblique_groove, horiz_groove)
                lx *= fissure_scale
                lz *= fissure_scale

                wx = base_cx + lx
                wy = y
                wz = lz - 0.12

                if cos_p < -0.55:
                    dome_indent = ((abs(cos_p) - 0.55) / 0.45) ** 1.6
                    wy += 0.32 * dome_indent * (1.0 - 0.3 * (math.cos(theta) ** 2))

                if not is_right_lung:
                    notch_y = math.exp(-((wy + 0.55) / 0.85) ** 2)
                    notch_medial = max(0.0, -math.cos(theta - 0.35)) ** 1.5
                    wx += 0.58 * notch_y * notch_medial

                vertices[i * n_lon + j] = [wx, wy, wz]

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
        return (vertices, normals, indices, (0.94, 0.56, 0.64, 0.22))

    def _build_trachea_bronchi_and_alveoli(self):
        airway_meshes = []
        self.alveolar_centers = []

        airway_col = (0.86, 0.72, 0.74, 1.0)
        ring_col = (0.95, 0.90, 0.86, 1.0)
        alveoli_col = (0.96, 0.62, 0.68, 1.0)

        trachea_pts = [
            [0.00, 3.25, -0.38],
            [0.00, 2.55, -0.36],
            [0.00, 1.82, -0.34],
        ]
        trachea_curve = catmull_rom_spline(trachea_pts, num_samples=22)
        vt, nt, it = build_tube_mesh(trachea_curve, 0.21, radial_segments=18)
        airway_meshes.append((vt, nt, it, airway_col))

        for ry in np.linspace(1.95, 3.12, 9):
            ring_pts = [
                [0.00, ry - 0.03, -0.36],
                [0.00, ry + 0.03, -0.36],
            ]
            vr, nr, ir = build_tube_mesh(ring_pts, 0.228, radial_segments=16)
            airway_meshes.append((vr, nr, ir, ring_col))

        right_branches = [
            ([[0.00, 1.82, -0.34], [-0.65, 1.55, -0.32], [-1.35, 1.25, -0.28]], 0.16, 0.13, False),
            ([[-1.35, 1.25, -0.28], [-1.78, 1.62, -0.24], [-2.15, 1.98, -0.18]], 0.11, 0.055, True),
            ([[-1.78, 1.62, -0.24], [-2.28, 1.72,  0.22]], 0.08, 0.050, True),
            ([[-1.78, 1.62, -0.24], [-2.35, 1.58, -0.58]], 0.08, 0.050, True),
            ([[-1.35, 1.25, -0.28], [-2.45, 1.22,  0.05]], 0.09, 0.052, True),
            ([[-1.35, 1.25, -0.28], [-1.62, 0.68, -0.24], [-1.85, 0.05, -0.22]], 0.13, 0.10, False),
            ([[-1.62, 0.68, -0.24], [-2.12, 0.55,  0.18], [-2.58, 0.38,  0.42]], 0.09, 0.050, True),
            ([[-2.12, 0.55,  0.18], [-2.68, 0.65, -0.12]], 0.075, 0.048, True),
            ([[-1.62, 0.68, -0.24], [-2.05, 0.25,  0.32]], 0.08, 0.048, True),
            ([[-1.85, 0.05, -0.22], [-2.25, -0.52, -0.15], [-2.58, -1.08,  0.12]], 0.095, 0.052, True),
            ([[-2.25, -0.52, -0.15], [-2.78, -0.88, -0.35]], 0.08, 0.048, True),
            ([[-1.85, 0.05, -0.22], [-2.35, -0.45, -0.58], [-2.62, -1.02, -0.52]], 0.09, 0.050, True),
            ([[-1.85, 0.05, -0.22], [-1.95, -0.68,  0.18], [-2.08, -1.18,  0.28]], 0.085, 0.048, True),
            ([[-2.25, -0.52, -0.15], [-2.85, -0.32,  0.08]], 0.075, 0.046, True),
        ]

        left_branches = [
            ([[0.00, 1.82, -0.34], [0.72, 1.48, -0.32], [1.42, 1.15, -0.28]], 0.15, 0.12, False),
            ([[1.42, 1.15, -0.28], [1.82, 1.55, -0.24], [2.18, 1.95, -0.16]], 0.105, 0.052, True),
            ([[1.82, 1.55, -0.24], [2.32, 1.65,  0.22]], 0.08, 0.048, True),
            ([[1.82, 1.55, -0.24], [2.38, 1.52, -0.55]], 0.08, 0.048, True),
            ([[1.42, 1.15, -0.28], [2.48, 1.12,  0.08]], 0.085, 0.050, True),
            ([[1.42, 1.15, -0.28], [1.95, 0.58,  0.12], [2.48, 0.18,  0.38]], 0.09, 0.048, True),
            ([[1.95, 0.58,  0.12], [2.58, 0.52, -0.15]], 0.075, 0.046, True),
            ([[1.42, 1.15, -0.28], [1.75, 0.42, -0.26], [1.98, -0.22, -0.24]], 0.12, 0.095, False),
            ([[1.98, -0.22, -0.24], [2.32, -0.68, -0.12], [2.58, -1.12,  0.15]], 0.09, 0.050, True),
            ([[2.32, -0.68, -0.12], [2.75, -0.92, -0.35]], 0.08, 0.048, True),
            ([[1.98, -0.22, -0.24], [2.38, -0.58, -0.56], [2.58, -1.05, -0.50]], 0.085, 0.048, True),
            ([[1.98, -0.22, -0.24], [2.12, -0.78,  0.20], [2.25, -1.18,  0.30]], 0.08, 0.046, True),
            ([[2.32, -0.68, -0.12], [2.78, -0.35,  0.08]], 0.075, 0.045, True),
        ]

        for b_pts, r_start, r_end, has_alveoli in (right_branches + left_branches):
            curve = catmull_rom_spline(b_pts, num_samples=14)
            radii = np.linspace(r_start, r_end, 14)
            vb, nb, ib = build_tube_mesh(curve, radii, radial_segments=10)
            airway_meshes.append((vb, nb, ib, airway_col))

            if has_alveoli:
                tip = np.array(b_pts[-1], dtype=np.float32)
                self.alveolar_centers.append(tip)
                va, na, ia = build_sphere_cluster_mesh(tip, radius=0.115, sub_lobules=6)
                airway_meshes.append((va, na, ia, alveoli_col))

        return airway_meshes

    def _build_translucent_vessels(self):
        vessels = []
        artery_rgba = (0.95, 0.20, 0.24, 0.28)
        vein_rgba = (0.24, 0.45, 0.95, 0.28)
        pulm_art_rgba = (0.32, 0.48, 0.95, 0.28)
        pulm_vein_rgba = (0.95, 0.24, 0.30, 0.28)

        # 1. Ascending Aorta, Aortic Arch, Descending Aorta + Supra-aortic branches
        aorta_pts = [
            [ 0.18,  0.54,  0.02],
            [ 0.24,  1.12,  0.10],
            [ 0.12,  1.78,  0.06],
            [-0.14,  2.14, -0.14],
            [-0.42,  1.98, -0.48],
            [-0.48,  1.32, -0.76],
            [-0.42,  0.10, -0.84],
            [-0.38, -1.52, -0.82],
        ]
        aorta_curve = catmull_rom_spline(aorta_pts, num_samples=52)
        aorta_radii = np.interp(np.linspace(0, 1, 52), [0.0, 0.25, 0.5, 1.0], [0.30, 0.34, 0.31, 0.26])
        vo, no, io = build_tube_mesh(aorta_curve, aorta_radii, radial_segments=22)
        vessels.append((vo, no, io, artery_rgba))

        branches = [
            ([[ 0.08, 1.98,  0.02], [ 0.20, 2.45,  0.04], [ 0.30, 2.82,  0.06]], 0.12),
            ([[-0.10, 2.12, -0.10], [-0.08, 2.55, -0.10], [-0.06, 2.90, -0.10]], 0.095),
            ([[-0.28, 2.08, -0.28], [-0.35, 2.52, -0.30], [-0.42, 2.85, -0.32]], 0.095),
        ]
        for b_pts, b_rad in branches:
            bc = catmull_rom_spline(b_pts, num_samples=14)
            vb, nb, ib = build_tube_mesh(bc, b_rad, radial_segments=14)
            vessels.append((vb, nb, ib, artery_rgba))

        # 2. Pulmonary Artery Trunk & Left/Right Pulmonary Arteries + Intrapulmonary Branches
        pulm_trunk_pts = [
            [-0.22,  0.76,  0.42],
            [-0.24,  1.28,  0.34],
            [-0.16,  1.62,  0.14],
            [ 0.45,  1.54, -0.12],
            [ 1.35,  1.35, -0.20],
            [ 2.05,  1.55, -0.15],
        ]
        pulm_curve = catmull_rom_spline(pulm_trunk_pts, num_samples=38)
        pulm_radii = np.interp(np.linspace(0, 1, 38), [0.0, 0.45, 1.0], [0.28, 0.26, 0.11])
        vp, np_n, ip = build_tube_mesh(pulm_curve, pulm_radii, radial_segments=18)
        vessels.append((vp, np_n, ip, pulm_art_rgba))

        rpa_pts = [
            [-0.16, 1.60,  0.12],
            [-0.75, 1.50, -0.10],
            [-1.45, 1.32, -0.18],
            [-2.15, 1.65, -0.12],
        ]
        rpa_curve = catmull_rom_spline(rpa_pts, num_samples=26)
        rpa_rad = np.interp(np.linspace(0, 1, 26), [0.0, 0.5, 1.0], [0.22, 0.19, 0.09])
        vrpa, nrpa, irpa = build_tube_mesh(rpa_curve, rpa_rad, radial_segments=16)
        vessels.append((vrpa, nrpa, irpa, pulm_art_rgba))

        intra_pa = [
            ([[-1.45, 1.32, -0.18], [-1.95, 0.35, -0.10], [-2.45, -0.75,  0.05]], 0.14, 0.06),
            ([[ 1.35, 1.35, -0.20], [ 1.92, 0.32, -0.12], [ 2.42, -0.78,  0.08]], 0.14, 0.06),
        ]
        for pts, r0, r1 in intra_pa:
            c = catmull_rom_spline(pts, num_samples=18)
            v, n, i = build_tube_mesh(c, np.linspace(r0, r1, 18), radial_segments=12)
            vessels.append((v, n, i, pulm_art_rgba))

        # 3. Superior Vena Cava (SVC) & Inferior Vena Cava (IVC)
        svc_curve = catmull_rom_spline([
            [-0.96,  2.55, -0.10],
            [-0.96,  1.75, -0.05],
            [-0.95,  0.96,  0.02],
        ], num_samples=24)
        v_svc, n_svc, i_svc = build_tube_mesh(svc_curve, 0.25, radial_segments=20)
        vessels.append((v_svc, n_svc, i_svc, vein_rgba))

        ivc_curve = catmull_rom_spline([
            [-0.98, -1.68, -0.28],
            [-0.96, -0.55, -0.15],
            [-0.95,  0.46,  0.00],
        ], num_samples=24)
        v_ivc, n_ivc, i_ivc = build_tube_mesh(ivc_curve, 0.26, radial_segments=20)
        vessels.append((v_ivc, n_ivc, i_ivc, vein_rgba))

        # 4. Four Pulmonary Veins entering Left Atrium
        pv_specs = [
            ([[ 2.25, 1.35, -0.15], [ 1.48, 1.02, -0.30], [ 0.80, 0.88, -0.28]], 0.07, 0.14),
            ([[ 2.38, -0.55, 0.05], [ 1.50, 0.35, -0.28], [ 0.78, 0.68, -0.28]], 0.07, 0.14),
            ([[-2.25, 1.25, -0.20], [-0.55, 1.00, -0.48], [ 0.68, 0.88, -0.32]], 0.07, 0.13),
            ([[-2.35, -0.55, 0.02], [-0.45, 0.38, -0.48], [ 0.68, 0.68, -0.32]], 0.07, 0.13),
        ]
        for pv_pts, r0, r1 in pv_specs:
            pvc = catmull_rom_spline(pv_pts, num_samples=22)
            vpv, npv, ipv = build_tube_mesh(pvc, np.linspace(r0, r1, 22), radial_segments=14)
            vessels.append((vpv, npv, ipv, pulm_vein_rgba))

        return vessels

    def _build_all_geometry(self):
        # 1A. Solid Opaque Inner Cardiac Core (Crux Septum + Trabeculae + Moderator Band + Papillary + Coronaries)
        heart_solid_list = self._build_solid_core_skeleton()
        self.heart_solid_batch = FastDeformableBatch(heart_solid_list, is_translucent=False)

        # 1B. Solid Opaque Inner Lung Airway Tree (Trachea, Bronchi, Alveoli)
        lung_solid_list = self._build_trachea_bronchi_and_alveoli()
        self.lung_solid_batch = FastDeformableBatch(lung_solid_list, is_translucent=False)

        # 2. Fresnel X-Ray Shaded Great Vessels Batch
        self.vessel_batch = FastDeformableBatch(self._build_translucent_vessels(), is_translucent=True)

        # 3A. Fresnel X-Ray Shaded 4 Endocardial Chambers (RA, LA, RV, LV) + Outer Epicardial Shell
        heart_shells = self._build_four_endocardial_chambers()
        heart_shells.append(self._build_outer_epicardial_shell())
        self.heart_shell_batch = FastDeformableBatch(heart_shells, is_translucent=True)

        # 3B. Fresnel X-Ray Shaded Right & Left Lung Shells
        lung_shells = [
            self._build_lung_shell(is_right_lung=True),
            self._build_lung_shell(is_right_lung=False),
        ]
        self.lung_shell_batch = FastDeformableBatch(lung_shells, is_translucent=True)


# =============================================================================
# 4 MULTI-LEAFLET HEART VALVES & DYNAMIC CHORDAE TENDINEAE
# =============================================================================

class HeartValveSystem:
    """
    Renders and animates all 4 solid cardiac valves and dynamic chordae tendineae
    aligned inside the 4 chamber orifices.
    """
    def __init__(self):
        self.valves = {
            "tricuspid": {
                "pos": np.array([-0.64,  0.22,  0.14], dtype=np.float32),
                "normal": normalize(np.array([0.14, -0.96, 0.12], dtype=np.float32)),
                "radius": 0.29,
                "num_leaflets": 3,
            },
            "mitral": {
                "pos": np.array([ 0.62,  0.24, -0.10], dtype=np.float32),
                "normal": normalize(np.array([0.08, -0.96, 0.14], dtype=np.float32)),
                "radius": 0.28,
                "num_leaflets": 2,
            },
            "pulmonary": {
                "pos": np.array([-0.22,  0.76,  0.42], dtype=np.float32),
                "normal": normalize(np.array([-0.04, 0.94, -0.22], dtype=np.float32)),
                "radius": 0.23,
                "num_leaflets": 3,
            },
            "aortic": {
                "pos": np.array([ 0.18,  0.54,  0.02], dtype=np.float32),
                "normal": normalize(np.array([0.10, 0.96, 0.14], dtype=np.float32)),
                "radius": 0.24,
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

    def render_valves_and_chordae(self, cycle: CardiopulmonaryEngine, papillary_anchors):
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
                for offset_scale in (-0.04, 0.0, 0.04):
                    cord_end = leaf_tip + np.array([offset_scale, 0.0, -offset_scale], dtype=np.float32)
                    mid_pt = 0.5 * (pap_tip + cord_end) + np.array([0.0, 0.02, 0.02], dtype=np.float32)
                    pts_c = ((1.0 - t_vals) ** 2) * pap_tip + 2.0 * (1.0 - t_vals) * t_vals * mid_pt + (t_vals ** 2) * cord_end
                    for k in range(len(pts_c) - 1):
                        glVertex3fv(pts_c[k])
                        glVertex3fv(pts_c[k + 1])
        glEnd()


# =============================================================================
# CLOSED-LOOP CARDIOPULMONARY BLOOD & AIRWAY GAS EXCHANGE PARTICLE SYSTEM
# =============================================================================

class CardiopulmonaryFlowSystem:
    """
    Simulates 520 3D Erythrocytes undergoing live alveolar gas exchange and
    220 respiratory O2/CO2 gas molecules oscillating in the bronchial tree.
    """
    def __init__(self, num_rbc=520, num_air=220):
        self.num_rbc = num_rbc
        self.num_air = num_air
        self._build_cardiopulmonary_splines()
        self._build_airway_splines()
        self._build_batched_rbc_template()
        self.reset_particles()

    def _build_cardiopulmonary_splines(self):
        c0 = [
            [-0.96,  2.45, -0.10],
            [-0.84,  0.72,  0.04],
            [-0.64,  0.22,  0.14],
            [-0.52, -0.88,  0.22],
            [-0.22,  0.76,  0.42],
            [-0.18,  1.52,  0.18],
            [ 1.35,  1.42, -0.18],
            [ 2.28,  1.65, -0.05],
            [ 1.50,  1.02, -0.30],
            [ 0.74,  0.74, -0.22],
            [ 0.62,  0.24, -0.10],
            [ 0.56, -1.32,  0.20],
            [ 0.18,  0.54,  0.02],
            [ 0.12,  1.82,  0.06],
            [-0.42,  1.82, -0.56],
            [-0.38, -1.42, -0.82],
        ]
        c1 = [
            [-0.98, -1.58, -0.28],
            [-0.84,  0.64,  0.04],
            [-0.64,  0.22,  0.14],
            [-0.45, -0.98,  0.24],
            [-0.22,  0.76,  0.42],
            [-0.18,  1.52,  0.18],
            [-1.42,  1.38, -0.18],
            [-2.35,  1.52, -0.08],
            [-0.55,  1.00, -0.48],
            [ 0.72,  0.74, -0.22],
            [ 0.62,  0.24, -0.10],
            [ 0.54, -1.35,  0.20],
            [ 0.18,  0.54,  0.02],
            [ 0.15,  1.88,  0.04],
            [-0.08,  2.32, -0.10],
            [-0.06,  2.88, -0.10],
        ]
        c2 = [
            [-0.96,  2.35, -0.10],
            [-0.84,  0.70,  0.04],
            [-0.64,  0.22,  0.14],
            [-0.54, -0.92,  0.22],
            [-0.22,  0.76,  0.42],
            [-0.16,  1.58,  0.16],
            [ 1.75,  0.48, -0.15],
            [ 2.48, -0.78,  0.08],
            [ 1.50,  0.35, -0.28],
            [ 0.74,  0.68, -0.22],
            [ 0.62,  0.24, -0.10],
            [ 0.56, -1.38,  0.22],
            [ 0.18,  0.54,  0.02],
            [ 0.10,  1.90,  0.04],
            [-0.45,  1.22, -0.76],
            [-0.38, -1.42, -0.82],
        ]
        c3 = [
            [-0.98, -1.58, -0.28],
            [-0.84,  0.66,  0.04],
            [-0.64,  0.22,  0.14],
            [-0.48, -0.95,  0.24],
            [-0.22,  0.76,  0.42],
            [-0.18,  1.56,  0.16],
            [-1.85,  0.42, -0.14],
            [-2.52, -0.75,  0.06],
            [-0.45,  0.38, -0.48],
            [ 0.72,  0.68, -0.22],
            [ 0.62,  0.24, -0.10],
            [ 0.52, -1.34,  0.20],
            [ 0.18,  0.54,  0.02],
            [ 0.14,  1.90,  0.03],
            [ 0.20,  2.45,  0.04],
            [ 0.30,  2.82,  0.06],
        ]

        self.n_samples = 260
        self.splines = np.stack([
            catmull_rom_spline(c0, num_samples=self.n_samples),
            catmull_rom_spline(c1, num_samples=self.n_samples),
            catmull_rom_spline(c2, num_samples=self.n_samples),
            catmull_rom_spline(c3, num_samples=self.n_samples),
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

    def _build_airway_splines(self):
        air_paths = [
            [[0.0, 3.22, -0.38], [0.0, 1.82, -0.34], [-1.35, 1.25, -0.28], [-2.15, 1.92, -0.18]],
            [[0.0, 3.22, -0.38], [0.0, 1.82, -0.34], [-1.35, 1.25, -0.28], [-1.62, 0.68, -0.24], [-2.52, 0.38,  0.38]],
            [[0.0, 3.22, -0.38], [0.0, 1.82, -0.34], [-1.35, 1.25, -0.28], [-1.85, 0.05, -0.22], [-2.55, -1.02, 0.10]],
            [[0.0, 3.22, -0.38], [0.0, 1.82, -0.34], [ 1.42, 1.15, -0.28], [ 2.15, 1.88, -0.16]],
            [[0.0, 3.22, -0.38], [0.0, 1.82, -0.34], [ 1.42, 1.15, -0.28], [ 1.95, 0.58,  0.12], [ 2.45, 0.18,  0.35]],
            [[0.0, 3.22, -0.38], [0.0, 1.82, -0.34], [ 1.42, 1.15, -0.28], [ 1.98, -0.22, -0.24], [ 2.55, -1.05, 0.12]],
        ]
        self.n_air_samples = 120
        self.air_splines = np.stack([
            catmull_rom_spline(ap, num_samples=self.n_air_samples) for ap in air_paths
        ], axis=0).astype(np.float32)

    def _build_batched_rbc_template(self):
        n_lat, n_lon = 6, 10
        radius = 0.044
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
        self.n_v_rbc = len(verts)

        self.batched_norms = np.tile(norms, (self.num_rbc, 1)).astype(np.float32)
        offsets = (np.arange(self.num_rbc, dtype=np.uint32) * self.n_v_rbc)[:, None]
        self.batched_indices = (indices[None, :] + offsets).ravel()
        self.batched_verts = np.zeros((self.num_rbc, self.n_v_rbc, 3), dtype=np.float32)
        self.batched_colors = np.zeros((self.num_rbc, self.n_v_rbc, 3), dtype=np.float32)

        self.streak_verts = np.zeros((self.num_rbc, 2, 3), dtype=np.float32)
        self.streak_colors = np.zeros((self.num_rbc, 2, 3), dtype=np.float32)

    def reset_particles(self):
        rng = np.random.default_rng(42)
        self.path_id = rng.integers(0, 4, size=self.num_rbc)
        self.s = rng.uniform(0.0, 1.0, size=self.num_rbc).astype(np.float32)
        self.radial_angle = rng.uniform(0.0, 2.0 * math.pi, size=self.num_rbc).astype(np.float32)
        self.radial_dist = rng.uniform(0.01, 0.08, size=self.num_rbc).astype(np.float32)
        self.speed_mult = rng.uniform(0.88, 1.14, size=self.num_rbc).astype(np.float32)
        self.current_vel = np.full(self.num_rbc, 0.18, dtype=np.float32)

        self.air_path_id = rng.integers(0, 6, size=self.num_air)
        self.air_s = rng.uniform(0.02, 0.98, size=self.num_air).astype(np.float32)
        self.air_jitter = rng.uniform(-0.045, 0.045, size=(self.num_air, 3)).astype(np.float32)

    def update(self, dt, cycle: CardiopulmonaryEngine):
        if dt <= 0.0:
            return

        s = self.s
        tric_open = cycle.tricuspid_open
        pulm_open = cycle.pulmonary_open
        mitr_open = cycle.mitral_open
        aort_open = cycle.aortic_open

        vel = np.full_like(s, 0.18)
        swirl = np.full_like(s, 1.8)

        m_ra = (s >= 0.08) & (s < 0.16)
        vel[m_ra] = 0.18 + 0.35 * max(0.0, cycle.atrial_contraction) if tric_open > 0.15 else 0.03
        if tric_open <= 0.15:
            swirl[m_ra] += 4.5

        m_rv = (s >= 0.16) & (s < 0.31)
        vel[m_rv] = 0.22 + 0.52 * cycle.ventricular_contraction if pulm_open > 0.15 else (0.16 * tric_open + 0.04)

        m_lungs = (s >= 0.31) & (s < 0.64)
        vel[m_lungs] = 0.16 + 0.28 * cycle.arterial_pulse + 0.06 * cycle.lung_inflation

        m_la = (s >= 0.64) & (s < 0.70)
        vel[m_la] = 0.18 + 0.35 * max(0.0, cycle.atrial_contraction) if mitr_open > 0.15 else 0.03
        if mitr_open <= 0.15:
            swirl[m_la] += 4.5

        m_lv = (s >= 0.70) & (s < 0.82)
        vel[m_lv] = 0.24 + 0.56 * cycle.ventricular_contraction if aort_open > 0.15 else (0.16 * mitr_open + 0.04)

        m_ao = s >= 0.82
        vel[m_ao] = 0.16 + 0.55 * cycle.arterial_pulse

        vel *= self.speed_mult
        s_next = s + vel * dt

        for gate_s, v_open in ((0.16, tric_open), (0.31, pulm_open), (0.70, mitr_open), (0.82, aort_open)):
            if v_open < 0.20:
                blocked = (s < gate_s) & (s_next >= gate_s)
                s_next[blocked] = gate_s - 0.003
                vel[blocked] = 0.02

        self.s = np.mod(s_next, 1.0)
        self.current_vel = vel
        self.radial_angle += swirl * dt

        air_vel = 0.52 * cycle.airflow_velocity
        self.air_s = np.clip(self.air_s + air_vel * dt, 0.02, 0.98)
        if cycle.airflow_velocity > 0.15:
            wrapped = self.air_s >= 0.97
            self.air_s[wrapped] = np.random.uniform(0.02, 0.12, size=np.count_nonzero(wrapped))
        elif cycle.airflow_velocity < -0.15:
            wrapped = self.air_s <= 0.03
            self.air_s[wrapped] = np.random.uniform(0.88, 0.97, size=np.count_nonzero(wrapped))

    def render(self, cycle: CardiopulmonaryEngine, show_lungs=True):
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

        oxy_frac = smoothstep01((self.s - 0.43) / 0.13)[:, None]
        venous_rgb = np.array([0.14, 0.46, 0.98], dtype=np.float32)
        arterial_rgb = np.array([0.98, 0.14, 0.18], dtype=np.float32)
        rbc_rgb = (1.0 - oxy_frac) * venous_rgb + oxy_frac * arterial_rgb

        self.batched_colors[:] = rbc_rgb[:, None, :]
        self.batched_verts[:] = world_pts[:, None, :] + self.rbc_base_verts[None, :, :]

        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.65, 0.65, 0.65, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 55.0)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, self.batched_verts.reshape(-1, 3))
        glNormalPointer(GL_FLOAT, 0, self.batched_norms)
        glColorPointer(3, GL_FLOAT, 0, self.batched_colors.reshape(-1, 3))
        glDrawElements(GL_TRIANGLES, len(self.batched_indices), GL_UNSIGNED_INT, self.batched_indices)

        glDisableClientState(GL_NORMAL_ARRAY)

        streak_len = np.minimum(0.15, 0.22 * self.current_vel)[:, None]
        self.streak_verts[:, 0, :] = world_pts
        self.streak_verts[:, 1, :] = world_pts - streak_len * tangents
        self.streak_colors[:, 0, :] = np.minimum(1.0, rbc_rgb + 0.15)
        self.streak_colors[:, 1, :] = rbc_rgb * 0.35

        glDisable(GL_LIGHTING)
        glLineWidth(1.8)
        glVertexPointer(3, GL_FLOAT, 0, self.streak_verts.reshape(-1, 3))
        glColorPointer(3, GL_FLOAT, 0, self.streak_colors.reshape(-1, 3))
        glDrawArrays(GL_LINES, 0, self.num_rbc * 2)

        if show_lungs:
            a_idx_f = self.air_s * (self.n_air_samples - 1)
            a_idx0 = a_idx_f.astype(np.int32)
            a_idx1 = np.minimum(a_idx0 + 1, self.n_air_samples - 1)
            a_frac = (a_idx_f - a_idx0)[:, None]

            air_pts = (
                (1.0 - a_frac) * self.air_splines[self.air_path_id, a_idx0]
                + a_frac * self.air_splines[self.air_path_id, a_idx1]
                + self.air_jitter
            )
            air_pts = deform_points_direct(air_pts, cycle)

            inhale_weight = float(np.clip(0.5 + 0.5 * cycle.airflow_velocity, 0.0, 1.0))
            air_col = (
                0.55 + 0.35 * inhale_weight,
                0.78 + 0.20 * inhale_weight,
                1.0,
            )
            glPointSize(4.5)
            glColor3fv(air_col)
            glDisableClientState(GL_COLOR_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, air_pts)
            glDrawArrays(GL_POINTS, 0, self.num_air)
        else:
            glDisableClientState(GL_COLOR_ARRAY)

        glEnable(GL_LIGHTING)
        glDisableClientState(GL_VERTEX_ARRAY)


# =============================================================================
# 2D HUD OVERLAY, ECG, PCG, SPIROMETRY & INTERACTIVE CONTROLS
# =============================================================================

class HUDOverlay:
    """Renders crisp 2D telemetry, ECG, PCG, Spirometry waveforms, and interactive controls."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        pygame.font.init()
        self.font_title = pygame.font.SysFont("Consolas", 15, bold=True)
        self.font_body = pygame.font.SysFont("Consolas", 12, bold=False)
        self.font_btn = pygame.font.SysFont("Consolas", 14, bold=True)
        self.tex_cache = {}

        y_btn = height - 52
        self.btn_pause_rect = pygame.Rect(18, y_btn, 148, 38)
        self.btn_reset_rect = pygame.Rect(176, y_btn, 148, 38)
        self.btn_slower_rect = pygame.Rect(338, y_btn, 42, 38)
        self.slider_rect = pygame.Rect(388, y_btn, 212, 38)
        self.btn_faster_rect = pygame.Rect(608, y_btn, 42, 38)
        self.btn_mute_rect = pygame.Rect(664, y_btn, 175, 38)
        self.btn_lungs_rect = pygame.Rect(849, y_btn, 175, 38)
        self.btn_hud_rect = pygame.Rect(1034, y_btn, 165, 38)

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

        if len(self.tex_cache) > 220:
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

    def render(
        self, cycle: CardiopulmonaryEngine, audio: CardiacAudioEngine,
        cam_rot, sim_speed, paused, fps, mouse_pos,
        show_lungs=True, show_hud=True
    ):
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

        if show_hud:
            # 1. Left Telemetry & Controls Panel
            self.draw_box(16, 14, 455, 310, (0.05, 0.08, 0.14, 0.78), (0.28, 0.45, 0.68, 0.85))
            self.draw_text(26, 22, "3D CARDIOPULMONARY SYSTEM (HEART & LUNGS)", self.font_title, (255, 115, 130))
            self.draw_text(26, 44, f"Cardiac Phase : {cycle.phase_name}", self.font_body, (230, 240, 255))
            self.draw_text(26, 61, f"Lung Phase    : {cycle.resp_phase_name}", self.font_body, (165, 245, 225))
            eff_bpm = cycle.bpm * sim_speed
            eff_rr = cycle.resp_rate * sim_speed
            self.draw_text(
                26, 78,
                f"Sim Speed     : {sim_speed:.2f}x ({eff_bpm:.0f} BPM | {eff_rr:.1f} Breaths/m | {fps:.0f} FPS)",
                self.font_body, (130, 235, 255)
            )

            snd_status = "MUTED" if not audio.enabled else audio.active_sound_label
            snd_col = (255, 130, 130) if not audio.enabled else (255, 230, 120)
            self.draw_text(26, 95, f"Auscultation  : {snd_status}", self.font_body, snd_col)

            rx, ry, rz = cam_rot
            self.draw_text(26, 113, f"Camera Angles : X={rx:5.1f}° | Y={ry:5.1f}° | Z={rz:5.1f}°", self.font_body, (160, 250, 195))
            self.draw_text(26, 134, "Controls (Full 3-Axis Camera + Speed + Audio):", self.font_body, (255, 215, 130))
            self.draw_text(26, 151, "  [UP / DOWN]          : Rotate X-Axis (Pitch)", self.font_body, (210, 220, 235))
            self.draw_text(26, 168, "  [LEFT / RIGHT]       : Rotate Y-Axis (Yaw)", self.font_body, (210, 220, 235))
            self.draw_text(26, 185, "  [SHIFT+LEFT / RIGHT] : Rotate Z-Axis (Roll) [or Q/E]", self.font_body, (210, 220, 235))
            self.draw_text(26, 202, "  [W / S] or Scroll    : Zoom Camera In / Out", self.font_body, (210, 220, 235))
            self.draw_text(26, 219, "  [ [ / ] ] or [1 - 5] : Simulation Speed (0.10x - 3.00x)", self.font_body, (155, 235, 255))
            self.draw_text(26, 236, "  [P] Pause | [R] Reset | [M] Sound | [L] Lungs | [H] HUD", self.font_body, (255, 225, 155))
            self.draw_text(26, 256, "Blood & Gas: [Blue] Venous -> [Alveoli O2] -> [Red] Arterial", self.font_body, (200, 215, 250))
            self.draw_text(26, 273, "Chambers: Sculpted RA, LA, RV, LV + Solid Crux Septum Core", self.font_body, (175, 240, 255))
            self.draw_text(26, 290, "Optics: Fresnel Rim-Lit Contours + Clear Center X-Ray Window", self.font_body, (175, 235, 195))

            # 2. Right Panel: Valves, Lead II ECG, Heart PCG & Lung Spirometry
            rx_box = self.width - 340
            self.draw_box(rx_box, 14, 324, 376, (0.05, 0.08, 0.14, 0.78), (0.28, 0.45, 0.68, 0.85))
            self.draw_text(rx_box + 14, 22, "VALVES, ECG, PCG & SPIROMETRY", self.font_title, (130, 230, 255))

            valves_info = [
                ("Tricuspid (RA->RV)", cycle.tricuspid_open),
                ("Mitral    (LA->LV)", cycle.mitral_open),
                ("Pulmonary (RV->PA)", cycle.pulmonary_open),
                ("Aortic    (LV->AO)", cycle.aortic_open),
            ]
            for idx, (v_label, v_open) in enumerate(valves_info):
                vy = 48 + idx * 22
                state_txt = "OPEN  " if v_open > 0.25 else "CLOSED"
                col = (110, 250, 145) if v_open > 0.25 else (255, 115, 115)
                self.draw_text(rx_box + 14, vy, f"{v_label}: {state_txt}", self.font_body, col)
                bar_x = rx_box + 225
                self.draw_box(bar_x, vy + 2, 82, 11, (0.18, 0.22, 0.28, 0.9))
                fill_col = (0.25, 0.92, 0.48, 0.95) if v_open > 0.25 else (0.92, 0.28, 0.28, 0.85)
                self.draw_box(bar_x, vy + 2, max(3.0, 82.0 * v_open), 11, fill_col)

            # Trace 1: Lead II ECG
            ecg_x, ecg_y, ecg_w, ecg_h = rx_box + 14, 142, 296, 68
            self.draw_box(ecg_x, ecg_y, ecg_w, ecg_h, (0.02, 0.05, 0.08, 0.92), (0.18, 0.38, 0.42, 0.8))
            self.draw_text(ecg_x + 6, ecg_y + 4, "Lead II ECG (Cardiac Electrical)", self.font_body, (110, 245, 165))
            glColor4f(0.20, 0.98, 0.48, 1.0)
            glLineWidth(1.9)
            glBegin(GL_LINE_STRIP)
            n_pts = len(cycle.ecg_history)
            for i, val in enumerate(cycle.ecg_history):
                px = ecg_x + (i / max(1, n_pts - 1)) * ecg_w
                py = (ecg_y + ecg_h * 0.68) - val * (ecg_h * 0.45)
                glVertex2f(px, py)
            glEnd()

            # Trace 2: Phonocardiogram (PCG)
            pcg_x, pcg_y, pcg_w, pcg_h = rx_box + 14, 218, 296, 68
            self.draw_box(pcg_x, pcg_y, pcg_w, pcg_h, (0.02, 0.05, 0.08, 0.92), (0.42, 0.35, 0.18, 0.8))
            self.draw_text(pcg_x + 6, pcg_y + 4, "Phonocardiogram (S1 Lub / S2 Dub)", self.font_body, (255, 220, 120))
            glColor4f(1.0, 0.78, 0.25, 1.0)
            glLineWidth(1.8)
            glBegin(GL_LINE_STRIP)
            n_pcg = len(cycle.pcg_history)
            for i, val in enumerate(cycle.pcg_history):
                px = pcg_x + (i / max(1, n_pcg - 1)) * pcg_w
                py = (pcg_y + pcg_h * 0.56) - val * (pcg_h * 0.36)
                glVertex2f(px, py)
            glEnd()

            # Trace 3: Pulmonary Spirometry
            sp_x, sp_y, sp_w, sp_h = rx_box + 14, 294, 296, 82
            self.draw_box(sp_x, sp_y, sp_w, sp_h, (0.02, 0.05, 0.08, 0.92), (0.22, 0.42, 0.58, 0.8))
            self.draw_text(
                sp_x + 6, sp_y + 4,
                f"Spirometry Tidal Vol ({cycle.lung_inflation*100:04.1f}% | O2 Sat: {cycle.alveolar_oxygen*100:04.1f}%)",
                self.font_body, (135, 235, 255)
            )
            glColor4f(0.32, 0.88, 1.0, 1.0)
            glLineWidth(2.0)
            glBegin(GL_LINE_STRIP)
            n_sp = len(cycle.resp_history)
            for i, val in enumerate(cycle.resp_history):
                px = sp_x + (i / max(1, n_sp - 1)) * sp_w
                py = (sp_y + sp_h - 10) - val * (sp_h - 30)
                glVertex2f(px, py)
            glEnd()

            # 3. Bottom Bar Standard Controls (shown when HUD is active)
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

            self.draw_box(*self.btn_mute_rect, m_fill, (0.55, 0.78, 1.0, 0.95))
            m_label = "[M] SOUND: ON" if audio.enabled else "[M] SOUND: OFF"
            self.draw_text(self.btn_mute_rect.x + 22, self.btn_mute_rect.y + 11, m_label, self.font_btn, (255, 255, 255))

        # 4. Always-accessible [L] LUNGS and [H] HUD Toggle Buttons
        lungs_hover = self.btn_lungs_rect.collidepoint(mouse_pos)
        hud_hover = self.btn_hud_rect.collidepoint(mouse_pos)

        l_fill = (0.16, 0.46, 0.42, 0.92) if show_lungs else (0.45, 0.22, 0.32, 0.92)
        if lungs_hover:
            l_fill = (min(1.0, l_fill[0] + 0.12), min(1.0, l_fill[1] + 0.12), min(1.0, l_fill[2] + 0.12), 0.95)

        h_fill = (0.14, 0.34, 0.56, 0.92) if show_hud else (0.22, 0.28, 0.38, 0.75)
        if hud_hover:
            h_fill = (min(1.0, h_fill[0] + 0.14), min(1.0, h_fill[1] + 0.14), min(1.0, h_fill[2] + 0.14), 0.95)

        self.draw_box(*self.btn_lungs_rect, l_fill, (0.55, 0.78, 1.0, 0.95))
        l_label = "[L] LUNGS: ON" if show_lungs else "[L] LUNGS: OFF"
        self.draw_text(self.btn_lungs_rect.x + 20, self.btn_lungs_rect.y + 11, l_label, self.font_btn, (255, 255, 255))

        self.draw_box(*self.btn_hud_rect, h_fill, (0.55, 0.78, 1.0, 0.95))
        h_label = "[H] HIDE HUD" if show_hud else "[H] SHOW HUD"
        self.draw_text(self.btn_hud_rect.x + 20, self.btn_hud_rect.y + 11, h_label, self.font_btn, (255, 255, 255))

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
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
    pygame.init()
    width, height = 1280, 720
    try:
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL, vsync=1)
    except TypeError:
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)

    pygame.display.set_caption(
        "PyOpenGL 3D Cardiopulmonary Simulator - 4-Chamber Fresnel X-Ray & Solid Crux Core"
    )

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glEnable(GL_NORMALIZE)
    glShadeModel(GL_SMOOTH)
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)

    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [5.5, 6.0, 7.5, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.95, 0.92, 0.90, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.70, 0.70, 0.70, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.25, 0.22, 0.26, 1.0])

    glEnable(GL_LIGHT1)
    glLightfv(GL_LIGHT1, GL_POSITION, [-6.5, 3.0, -5.5, 1.0])
    glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.42, 0.50, 0.68, 1.0])
    glLightfv(GL_LIGHT1, GL_SPECULAR, [0.28, 0.32, 0.42, 1.0])

    glEnable(GL_LIGHT2)
    glLightfv(GL_LIGHT2, GL_POSITION, [0.0, -7.0, 3.5, 1.0])
    glLightfv(GL_LIGHT2, GL_DIFFUSE, [0.28, 0.22, 0.25, 1.0])

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(42.0, width / float(height), 0.1, 60.0)
    glMatrixMode(GL_MODELVIEW)

    # Instantiate Synchronized Cardiopulmonary Systems
    cycle = CardiopulmonaryEngine(bpm=72.0, resp_rate=18.0)
    audio = CardiacAudioEngine(sample_rate=44100)
    anatomy = CardiopulmonaryAnatomyModel()
    valves = HeartValveSystem()
    flow = CardiopulmonaryFlowSystem(num_rbc=520, num_air=220)
    hud = HUDOverlay(width, height)

    init_cam_rot = [8.0, -12.0, 0.0]
    cam_rot = list(init_cam_rot)
    cam_zoom = 9.6
    sim_speed = 1.0
    paused = False
    show_lungs = True
    show_hud = True
    dragging_slider = False

    def reset_simulation():
        nonlocal cam_rot, cam_zoom, sim_speed, paused, show_lungs, show_hud
        cam_rot = list(init_cam_rot)
        cam_zoom = 9.6
        sim_speed = 1.0
        paused = False
        show_lungs = True
        show_hud = True
        cycle.reset()
        flow.reset_particles()

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
                elif event.key == K_l:
                    show_lungs = not show_lungs
                elif event.key == K_h:
                    show_hud = not show_hud
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
                    if hud.btn_lungs_rect.collidepoint(event.pos):
                        show_lungs = not show_lungs
                    elif hud.btn_hud_rect.collidepoint(event.pos):
                        show_hud = not show_hud
                    elif show_hud:
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
                    cam_zoom = max(3.5, cam_zoom - 0.45)
                elif event.button == 5:
                    cam_zoom = min(18.0, cam_zoom + 0.45)
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_slider = False
            elif event.type == MOUSEMOTION:
                if dragging_slider and show_hud:
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
            cam_zoom = max(3.5, cam_zoom - 4.5 * dt)
        if keys[K_s] or keys[K_MINUS]:
            cam_zoom = min(18.0, cam_zoom + 4.5 * dt)

        cam_view_dir = compute_model_space_camera_dir(cam_rot)

        sim_dt = 0.0 if paused else (dt * sim_speed)
        cycle.update(sim_dt)
        audio.update_and_trigger(
            cycle.prev_phase, cycle.phase,
            cycle.prev_resp_phase, cycle.resp_phase,
            dt, paused
        )
        flow.update(sim_dt, cycle)

        anatomy.heart_solid_batch.update_geometry(cycle, cam_view_dir)
        anatomy.vessel_batch.update_geometry(cycle, cam_view_dir)
        anatomy.heart_shell_batch.update_geometry(cycle, cam_view_dir)
        if show_lungs:
            anatomy.lung_solid_batch.update_geometry(cycle, cam_view_dir)
            anatomy.lung_shell_batch.update_geometry(cycle, cam_view_dir)

        glClearColor(0.035, 0.045, 0.075, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        glTranslatef(0.0, -0.20, -cam_zoom)
        glRotatef(cam_rot[0], 1.0, 0.0, 0.0)
        glRotatef(cam_rot[1], 0.0, 1.0, 0.0)
        glRotatef(cam_rot[2], 0.0, 0.0, 1.0)

        # =====================================================================
        # PASS 1: OPAQUE SOLID CORE SKELETON, VALVES & BLOOD
        # =====================================================================
        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.40, 0.34, 0.34, 1.0])

        anatomy.heart_solid_batch.draw_opaque(shininess=32.0)
        if show_lungs:
            anatomy.lung_solid_batch.draw_opaque(shininess=32.0)
        valves.render_valves_and_chordae(cycle, anatomy.papillary_anchors)
        flow.render(cycle, show_lungs=show_lungs)

        # =====================================================================
        # PASS 2: FRESNEL X-RAY SHADED 4 CHAMBERS, OUTER HEART SHELL & LUNGS
        #   - Center of camera-facing walls = clear low-alpha window
        #   - Edges, chamber boundaries & wall thickness = bold & rich!
        # =====================================================================
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        glEnable(GL_CULL_FACE)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.45, 0.40, 0.42, 1.0])

        # Pass 2A: Backside surfaces (rich anatomical chamber backdrop)
        glCullFace(GL_FRONT)
        anatomy.vessel_batch.draw_translucent_pass(is_front_pass=False, shininess=45.0)
        anatomy.heart_shell_batch.draw_translucent_pass(is_front_pass=False, shininess=58.0)
        if show_lungs:
            anatomy.lung_shell_batch.draw_translucent_pass(is_front_pass=False, shininess=55.0)

        # Pass 2B: Frontside camera-facing surfaces (rim-lit contours + clear center window)
        glCullFace(GL_BACK)
        anatomy.vessel_batch.draw_translucent_pass(is_front_pass=True, shininess=45.0)
        anatomy.heart_shell_batch.draw_translucent_pass(is_front_pass=True, shininess=58.0)
        if show_lungs:
            anatomy.lung_shell_batch.draw_translucent_pass(is_front_pass=True, shininess=55.0)

        glDisable(GL_CULL_FACE)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

        # =====================================================================
        # PASS 3: 2D HUD TELEMETRY, ECG, PCG, SPIROMETRY & BUTTONS
        # =====================================================================
        hud.render(
            cycle, audio, cam_rot, sim_speed, paused, fps_smooth, mouse_pos,
            show_lungs=show_lungs, show_hud=show_hud
        )

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()