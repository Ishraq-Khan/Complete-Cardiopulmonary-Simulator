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


def catmull_rom_spline(points, num_samples=60):
    """Evaluates a smooth 3D Catmull-Rom spline through control points."""
    pts = np.asarray(points, dtype=np.float32)
    n = len(pts)
    if n < 2:
        return pts

    # Pad endpoints for full curve coverage
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

    # Parallel transport frame computation
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
            # Rodrigues rotation of previous normal
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
# PHYSIOLOGICAL CARDIAC CYCLE ENGINE
# =============================================================================

class CardiacCycleEngine:
    """
    Computes physiological parameters of the human cardiac cycle (75 BPM, T = 0.8s):
      - Atrial Systole (0.00 - 0.14): Atria contract, AV valves wide open.
      - Isovolumetric Contraction (0.14 - 0.19): AV valves snap shut, ventricles tense.
      - Ventricular Ejection / Systole (0.19 - 0.48): Semilunar valves open, ventricles contract/twist.
      - Isovolumetric Relaxation (0.48 - 0.54): Semilunar valves snap shut (dicrotic notch).
      - Passive Ventricular Filling / Diastole (0.54 - 1.00): AV valves open, chambers refill.
    """
    def __init__(self, bpm=72.0):
        self.bpm = bpm
        self.period = 60.0 / bpm
        self.time = 0.0
        self.phase = 0.0

        # Dynamic anatomical states
        self.atrial_contraction = 0.0       # [0..1]
        self.ventricular_contraction = 0.0  # [0..1]
        self.arterial_pulse = 0.0           # [0..1]
        self.apical_twist = 0.0             # radians

        # Valve openings [0 = completely closed, 1 = fully open]
        self.tricuspid_open = 1.0
        self.mitral_open = 1.0
        self.pulmonary_open = 0.0
        self.aortic_open = 0.0

        self.phase_name = "Diastole (Passive Filling)"
        self.ecg_history = [0.0] * 180

    def reset(self):
        self.time = 0.0
        self.ecg_history = [0.0] * 180
        self.update(0.0)

    def _compute_ecg(self, p):
        """Synthesizes a clinical P-QRS-T electrocardiogram waveform."""
        # P wave (atrial depolarization)
        p_wave = 0.18 * math.exp(-((p - 0.06) / 0.032) ** 2)
        # Q wave
        q_wave = -0.15 * math.exp(-((p - 0.155) / 0.010) ** 2)
        # R wave (ventricular depolarization peak)
        r_wave = 1.05 * math.exp(-((p - 0.175) / 0.012) ** 2)
        # S wave
        s_wave = -0.28 * math.exp(-((p - 0.195) / 0.011) ** 2)
        # T wave (ventricular repolarization)
        t_wave = 0.30 * math.exp(-((p - 0.41) / 0.048) ** 2)
        return p_wave + q_wave + r_wave + s_wave + t_wave

    def update(self, dt):
        self.time += dt
        self.phase = (self.time % self.period) / self.period
        p = self.phase

        # 1. Atrial Contraction Profile
        if p < 0.15:
            self.atrial_contraction = math.sin((p / 0.15) * math.pi) ** 1.3
        else:
            # Subtle expansion during atrial filling
            self.atrial_contraction = -0.08 * math.sin(((p - 0.15) / 0.85) * math.pi)

        # 2. Ventricular Contraction Profile
        if 0.14 <= p < 0.50:
            norm_v = (p - 0.14) / 0.36
            self.ventricular_contraction = math.sin(norm_v * math.pi) ** 1.15
        elif 0.50 <= p < 0.65:
            norm_r = (p - 0.50) / 0.15
            self.ventricular_contraction = 0.08 * (1.0 - norm_r)
        else:
            # Slight end-diastolic stretch during atrial kick
            if p < 0.14:
                self.ventricular_contraction = -0.07 * math.sin((p / 0.14) * math.pi)
            else:
                self.ventricular_contraction = 0.0

        # 3. Arterial Pulse Wave (Aorta & Pulmonary Trunk expansion)
        if 0.20 <= p < 0.62:
            norm_a = (p - 0.20) / 0.42
            # Includes dicrotic wave
            main_wave = math.sin(norm_a * math.pi) ** 1.4
            dicrotic = 0.18 * math.exp(-((p - 0.52) / 0.035) ** 2)
            self.arterial_pulse = min(1.0, main_wave + dicrotic)
        else:
            self.arterial_pulse = 0.0

        self.apical_twist = 0.22 * self.ventricular_contraction

        # 4. Atrioventricular Valves (Tricuspid & Mitral)
        if p < 0.13:
            # Atrial systole pushes AV valves maximally open
            av_open = 0.78 + 0.22 * math.sin((p / 0.13) * math.pi)
        elif 0.13 <= p < 0.17:
            # Rapid closure during onset of ventricular systole
            av_open = max(0.0, 1.0 - (p - 0.13) / 0.04)
        elif 0.17 <= p < 0.54:
            # Closed tightly throughout systole and isovolumetric relaxation
            av_open = 0.0
        elif 0.54 <= p < 0.60:
            # Rapid opening as ventricular pressure drops below atrial pressure
            av_open = (p - 0.54) / 0.06
        else:
            # Open during passive diastole (with mid-diastolic partial float)
            av_open = 0.78 + 0.12 * math.sin(((p - 0.60) / 0.40) * math.pi)

        self.tricuspid_open = float(np.clip(av_open, 0.0, 1.0))
        self.mitral_open = float(np.clip(av_open, 0.0, 1.0))

        # 5. Semilunar Valves (Pulmonary & Aortic)
        if 0.19 <= p < 0.23:
            # Rapid ejection opening
            sl_open = (p - 0.19) / 0.04
        elif 0.23 <= p < 0.45:
            # Wide open during ventricular ejection
            sl_open = 1.0 - 0.15 * ((p - 0.23) / 0.22) ** 2
        elif 0.45 <= p < 0.49:
            # Snap shut at end of ejection
            sl_open = max(0.0, 0.85 * (1.0 - (p - 0.45) / 0.04))
        else:
            sl_open = 0.0

        self.pulmonary_open = float(np.clip(sl_open, 0.0, 1.0))
        self.aortic_open = float(np.clip(sl_open, 0.0, 1.0))

        # Phase Name for HUD
        if p < 0.14:
            self.phase_name = "Atrial Systole (Atria Contract, AV Open)"
        elif p < 0.19:
            self.phase_name = "Isovolumetric Contraction (All Valves Closed)"
        elif p < 0.48:
            self.phase_name = "Ventricular Systole (Ejection, Semilunar Open)"
        elif p < 0.54:
            self.phase_name = "Isovolumetric Relaxation (All Valves Closed)"
        else:
            self.phase_name = "Ventricular Diastole (Passive Filling, AV Open)"

        if dt > 0:
            self.ecg_history.append(self._compute_ecg(p))
            if len(self.ecg_history) > 180:
                self.ecg_history.pop(0)


# =============================================================================
# DYNAMIC ANATOMICAL DEFORMATION FIELD
# =============================================================================

def deform_heart_vertices(base_verts, cycle: CardiacCycleEngine):
    """
    Applies spatially localized biomechanics to any 3D mesh in heart coordinates:
      - Ventricular region (y < 0.35): radial squeeze, longitudinal apex elevation, and physiological twist.
      - Atrial region (0.35 <= y < 1.35): radial contraction centered on Left/Right Atria.
      - Great Arterial Vessels (y >= 0.9): pulsatile systolic dilation.
    """
    v = base_verts.copy()
    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    # 1. Ventricular region mask (smoothly active for y < 0.35)
    v_mask = np.clip((0.40 - y) / 1.85, 0.0, 1.0)
    v_mask = v_mask * v_mask * (3.0 - 2.0 * v_mask)
    vc = cycle.ventricular_contraction

    # Ventricular radial contraction towards the interventricular axis
    vent_center_x = 0.12 * (0.3 - y)
    vent_center_z = 0.08
    dx_v = x - vent_center_x
    dz_v = z - vent_center_z

    squeeze = 1.0 - 0.24 * vc * v_mask
    x_new = vent_center_x + dx_v * squeeze
    z_new = vent_center_z + dz_v * squeeze
    # Apex moves superiorly during ventricular contraction
    y_new = y + 0.22 * vc * (v_mask ** 1.4)

    # Apical torsion (counter-clockwise twist at the apex)
    twist_angle = cycle.apical_twist * (v_mask ** 1.3)
    cos_t = np.cos(twist_angle)
    sin_t = np.sin(twist_angle)
    rx = x_new - vent_center_x
    rz = z_new - vent_center_z
    x_new = vent_center_x + rx * cos_t - rz * sin_t
    z_new = vent_center_z + rx * sin_t + rz * cos_t

    # 2. Atrial region mask (0.25 < y < 1.35)
    a_mask = np.exp(-((y - 0.78) / 0.42) ** 2)
    ac = cycle.atrial_contraction
    # Right Atrium center (-1.1, 0.75, 0.0) & Left Atrium center (+0.85, 0.80, -0.35)
    ra_dist = np.exp(-((x + 1.10) ** 2 + (z - 0.05) ** 2) / 0.95)
    la_dist = np.exp(-((x - 0.85) ** 2 + (z + 0.35) ** 2) / 0.95)

    atrial_pull_ra = 0.22 * ac * a_mask * ra_dist
    atrial_pull_la = 0.22 * ac * a_mask * la_dist

    x_new = x_new - atrial_pull_ra * (x_new + 1.10) - atrial_pull_la * (x_new - 0.85)
    y_new = y_new - 0.12 * ac * a_mask * (ra_dist + la_dist) * (y_new - 0.65)
    z_new = z_new - atrial_pull_ra * (z_new - 0.05) - atrial_pull_la * (z_new + 0.35)

    # 3. Arterial pulse wave (Aorta & Pulmonary Trunk dilation for y > 0.85)
    art_mask = np.clip((y - 0.85) / 1.2, 0.0, 1.0)
    ap = cycle.arterial_pulse
    x_new = x_new * (1.0 + 0.06 * ap * art_mask)
    z_new = z_new * (1.0 + 0.06 * ap * art_mask)

    v[:, 0] = x_new
    v[:, 1] = y_new
    v[:, 2] = z_new
    return v


# =============================================================================
# PROCEDURAL 3D ANATOMICAL HEART BUILDER
# =============================================================================

class AnatomicalHeartModel:
    """
    Constructs all solid internal structures and the semi-transparent outer shell:
      - Solid Inner Structures:
          1. Muscular Interventricular & Interatrial Septum
          2. Sculpted Endocardial Chamber Walls (RA, RV, LA, LV posterior/medial cradles)
          3. Trabeculae Carneae & 5 Conical Papillary Muscles
          4. Dynamic Chordae Tendineae connecting Papillary Muscles to AV Valve Leaflets
          5. 4 Fibrous Annuli & Multi-Leaflet Animated Valves (Tricuspid, Mitral, Pulmonary, Aortic)
          6. Coronary Artery & Cardiac Vein Network (LAD, Circumflex, RCA, Great Cardiac Vein)
      - Semi-Transparent Outer Layer:
          7. Bilobed Epicardial / Myocardial Outer Shell with Coronary & Interventricular Sulci
          8. Translucent Great Vessel Outer Sleeves (Aortic Arch + Branches, Pulmonary Trunk, SVC, IVC, PVs)
    """
    def __init__(self):
        self.solid_meshes = []
        self.vessel_inner_meshes = []
        self.coronary_meshes = []
        self.papillary_anchors = []
        self._build_all_geometry()

    def _build_outer_epicardial_shell(self):
        """Creates the detailed, anatomically contoured outer heart shell."""
        n_lat = 56
        n_lon = 72
        u = np.linspace(0.02, math.pi - 0.02, n_lat)
        v = np.linspace(0.0, 2.0 * math.pi, n_lon, endpoint=False)

        vertices = np.zeros((n_lat * n_lon, 3), dtype=np.float32)
        for i, phi in enumerate(u):
            # Normalized vertical coordinate from +1 (base) to -1 (apex)
            cos_p = math.cos(phi)
            sin_p = math.sin(phi)

            for j, theta in enumerate(v):
                # Base ovoid radius with realistic ventricular tapering toward the apex
                r_base = 1.48 * sin_p * (1.0 + 0.18 * cos_p - 0.12 * (cos_p ** 2))

                # Anatomical grooves (sulci)
                # 1. Anterior & Posterior Interventricular Sulci (groove between LV and RV)
                iv_groove_ant = -0.14 * math.exp(-((theta - 1.38) / 0.32) ** 2) * (sin_p ** 1.5)
                iv_groove_post = -0.11 * math.exp(-((theta - 4.55) / 0.35) ** 2) * (sin_p ** 1.5)

                # 2. Coronary (Atrioventricular) Sulcus separating Atria (top) and Ventricles (bottom)
                av_sulcus = -0.15 * math.exp(-((cos_p - 0.34) / 0.16) ** 2)

                # 3. Left & Right Atrial Auricle bulges (superior-lateral)
                ra_bulge = 0.28 * math.exp(-((cos_p - 0.58) / 0.28) ** 2) * math.exp(-((theta - 2.85) / 0.65) ** 2)
                la_bulge = 0.25 * math.exp(-((cos_p - 0.60) / 0.28) ** 2) * math.exp(-((theta - 5.80) / 0.65) ** 2)

                # 4. Left Ventricular muscular prominence
                lv_bulge = 0.22 * math.exp(-((cos_p + 0.18) / 0.55) ** 2) * math.exp(-((theta - 0.35) / 0.85) ** 2)

                r = r_base + iv_groove_ant + iv_groove_post + av_sulcus + ra_bulge + la_bulge + lv_bulge

                x = r * math.cos(theta) * 1.12
                z = r * math.sin(theta) * 0.86
                y = 1.38 * cos_p - 0.35

                # Anatomical tilt of the cardiac axis toward the left anterior apex
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

        # Close top and bottom poles cleanly
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
        return vertices, normals, indices

    def _build_interventricular_septum(self):
        """Builds the thick solid muscular wall dividing the Left and Right chambers."""
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
        # Flatten the septum in X and widen it in Z (anatomical blade profile)
        verts[:, 0] *= 0.52
        verts[:, 2] *= 1.65
        norms = compute_vertex_normals(verts, idx)
        return verts, norms, idx

    def _build_inner_chamber_cradles(self):
        """
        Builds solid sculpted inner endocardial walls & trabeculae ridges for RA, RV, LA, LV
        with open anterior/lateral viewing windows so the valves and blood remain clearly visible.
        """
        cradles = []

        # 1. Right Ventricle Solid Posterior/Inferior Muscular Cradle
        rv_path = catmull_rom_spline([
            [-0.90,  0.32,  0.12],
            [-0.95, -0.35,  0.18],
            [-0.65, -1.05,  0.22],
            [-0.12, -1.48,  0.28],
            [-0.22, -0.45,  0.42],
            [-0.25,  0.82,  0.52],
        ], num_samples=34)
        rv_radii = np.linspace(0.55, 0.36, 34)
        v_rv, n_rv, i_rv = build_tube_mesh(rv_path, rv_radii, radial_segments=20, arc_range=(2.2, 5.5))
        cradles.append((v_rv, n_rv, i_rv, (0.68, 0.18, 0.22)))

        # 2. Left Ventricle Solid Thick Myocardial Cradle
        lv_path = catmull_rom_spline([
            [ 0.78,  0.35, -0.12],
            [ 0.92, -0.35,  0.02],
            [ 0.88, -1.15,  0.15],
            [ 0.68, -1.72,  0.28],
            [ 0.38, -0.85,  0.12],
            [ 0.22,  0.58,  0.04],
        ], num_samples=36)
        lv_radii = np.linspace(0.58, 0.38, 36)
        v_lv, n_lv, i_lv = build_tube_mesh(lv_path, lv_radii, radial_segments=22, arc_range=(3.4, 6.9))
        cradles.append((v_lv, n_lv, i_lv, (0.76, 0.16, 0.20)))

        # 3. Right Atrium Solid Posterior/Pectinate Cradle
        ra_path = catmull_rom_spline([
            [-1.15,  1.25, -0.12],
            [-1.22,  0.75, -0.02],
            [-1.02,  0.30,  0.12],
        ], num_samples=20)
        ra_radii = np.array([0.42, 0.56, 0.48], dtype=np.float32)
        ra_radii = np.interp(np.linspace(0, 1, 20), [0.0, 0.5, 1.0], ra_radii)
        v_ra, n_ra, i_ra = build_tube_mesh(ra_path, ra_radii, radial_segments=18, arc_range=(1.8, 5.2))
        cradles.append((v_ra, n_ra, i_ra, (0.62, 0.20, 0.28)))

        # 4. Left Atrium Solid Posterior Cradle
        la_path = catmull_rom_spline([
            [ 0.88,  1.22, -0.45],
            [ 0.86,  0.78, -0.35],
            [ 0.78,  0.32, -0.15],
        ], num_samples=20)
        la_radii = np.interp(np.linspace(0, 1, 20), [0.0, 0.5, 1.0], [0.40, 0.54, 0.45])
        v_la, n_la, i_la = build_tube_mesh(la_path, la_radii, radial_segments=18, arc_range=(3.2, 6.6))
        cradles.append((v_la, n_la, i_la, (0.78, 0.22, 0.26)))

        # 5. Trabeculae Carneae (Solid inner muscular ridges inside RV and LV)
        trabeculae_specs = [
            ([[-0.85, -0.45, 0.02], [-0.55, -0.95, 0.08], [-0.15, -1.25, 0.18]], 0.075),
            ([[-0.72, -0.15, -0.08], [-0.42, -0.68, -0.02], [-0.08, -0.92, 0.12]], 0.068),
            ([[ 0.85, -0.38, -0.18], [ 0.78, -0.98, -0.05], [ 0.55, -1.48, 0.16]], 0.082),
            ([[ 0.55, -0.25, -0.26], [ 0.52, -0.82, -0.14], [ 0.38, -1.32, 0.08]], 0.075),
        ]
        for pts, rad in trabeculae_specs:
            curve = catmull_rom_spline(pts, num_samples=16)
            vt, nt, it = build_tube_mesh(curve, rad, radial_segments=10)
            cradles.append((vt, nt, it, (0.72, 0.15, 0.19)))

        return cradles

    def _build_papillary_muscles(self):
        """
        Builds 5 solid conical Papillary Muscles (3 in Right Ventricle, 2 in Left Ventricle)
        that anchor the Chordae Tendineae connected to the Tricuspid and Mitral valves.
        """
        specs = [
            # Right Ventricle (3 papillary muscles for Tricuspid valve)
            {"base": np.array([-0.92, -0.88,  0.32]), "tip": np.array([-0.82, -0.25,  0.24]), "valve": "tricuspid", "leaf_idx": 0},
            {"base": np.array([-0.68, -1.05, -0.05]), "tip": np.array([-0.74, -0.28,  0.04]), "valve": "tricuspid", "leaf_idx": 1},
            {"base": np.array([-0.22, -0.82,  0.16]), "tip": np.array([-0.48, -0.22,  0.16]), "valve": "tricuspid", "leaf_idx": 2},
            # Left Ventricle (2 large papillary muscles for Mitral valve)
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
            meshes.append((vp, np_norm, ip, (0.75, 0.18, 0.22)))
            self.papillary_anchors.append(sp)
        return meshes

    def _build_great_vessels(self):
        """
        Builds both the solid inner vascular necks/rings and the semi-transparent
        outer vascular walls for:
          - Ascending Aorta, Aortic Arch, Descending Aorta + 3 Arch Branches
          - Pulmonary Trunk + Left & Right Pulmonary Arteries
          - Superior Vena Cava (SVC) & Inferior Vena Cava (IVC)
          - 4 Pulmonary Veins (Left/Right Superior & Inferior)
        """
        outer_vessels = []
        inner_vessels = []

        # 1. Aorta (LV outflow -> Ascending -> Arch -> Descending)
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
        aorta_curve = catmull_rom_spline(aorta_pts, num_samples=56)
        aorta_radii = np.interp(np.linspace(0, 1, 56), [0.0, 0.25, 0.5, 1.0], [0.34, 0.38, 0.34, 0.28])
        vo, no, io = build_tube_mesh(aorta_curve, aorta_radii, radial_segments=24)
        outer_vessels.append((vo, no, io, (0.92, 0.18, 0.22, 0.36)))
        # Solid inner posterior lumen wall of the Aortic Arch so structure feels solid
        vi, ni, ii = build_tube_mesh(aorta_curve[:38], aorta_radii[:38] * 0.92, radial_segments=18, arc_range=(3.14, 6.28))
        inner_vessels.append((vi, ni, ii, (0.78, 0.14, 0.18)))

        # Supra-aortic branches (Brachiocephalic, Left Common Carotid, Left Subclavian)
        branches = [
            ([[ 0.10, 2.02,  0.04], [ 0.22, 2.48,  0.06], [ 0.32, 2.85,  0.08]], 0.13),
            ([[-0.08, 2.15, -0.08], [-0.06, 2.58, -0.08], [-0.04, 2.92, -0.08]], 0.10),
            ([[-0.28, 2.12, -0.28], [-0.35, 2.55, -0.30], [-0.42, 2.88, -0.32]], 0.10),
        ]
        for b_pts, b_rad in branches:
            bc = catmull_rom_spline(b_pts, num_samples=16)
            vb, nb, ib = build_tube_mesh(bc, b_rad, radial_segments=14)
            inner_vessels.append((vb, nb, ib, (0.85, 0.16, 0.20)))

        # 2. Pulmonary Artery Trunk & Bifurcation (Right & Left Pulmonary Arteries)
        pulm_trunk_pts = [
            [-0.25,  0.88,  0.52],
            [-0.28,  1.35,  0.42],
            [-0.18,  1.68,  0.18],
            [ 0.22,  1.72, -0.15],
            [ 0.95,  1.68, -0.32],
            [ 1.48,  1.62, -0.42],
        ]
        pulm_curve = catmull_rom_spline(pulm_trunk_pts, num_samples=40)
        pulm_radii = np.interp(np.linspace(0, 1, 40), [0.0, 0.45, 1.0], [0.32, 0.31, 0.22])
        vp, np_n, ip = build_tube_mesh(pulm_curve, pulm_radii, radial_segments=22)
        outer_vessels.append((vp, np_n, ip, (0.35, 0.42, 0.88, 0.36)))
        vpi, npi, ipi = build_tube_mesh(pulm_curve, pulm_radii * 0.92, radial_segments=16, arc_range=(3.14, 6.28))
        inner_vessels.append((vpi, npi, ipi, (0.28, 0.34, 0.72)))

        # Right Pulmonary Artery branch (going to anatomical right / -X)
        rpa_pts = [
            [-0.18, 1.65,  0.15],
            [-0.68, 1.62, -0.12],
            [-1.38, 1.56, -0.28],
        ]
        rpa_curve = catmull_rom_spline(rpa_pts, num_samples=22)
        vrpa, nrpa, irpa = build_tube_mesh(rpa_curve, 0.21, radial_segments=16)
        inner_vessels.append((vrpa, nrpa, irpa, (0.30, 0.36, 0.76)))

        # 3. Superior Vena Cava (SVC) & Inferior Vena Cava (IVC)
        svc_curve = catmull_rom_spline([
            [-1.12,  2.65, -0.15],
            [-1.12,  1.85, -0.10],
            [-1.12,  1.05, -0.02],
        ], num_samples=26)
        v_svc, n_svc, i_svc = build_tube_mesh(svc_curve, 0.28, radial_segments=20)
        outer_vessels.append((v_svc, n_svc, i_svc, (0.22, 0.38, 0.85, 0.36)))
        v_svci, n_svci, i_svci = build_tube_mesh(svc_curve, 0.26, radial_segments=16, arc_range=(3.0, 6.2))
        inner_vessels.append((v_svci, n_svci, i_svci, (0.20, 0.32, 0.70)))

        ivc_curve = catmull_rom_spline([
            [-1.12, -1.75, -0.35],
            [-1.12, -0.65, -0.22],
            [-1.10,  0.48, -0.05],
        ], num_samples=26)
        v_ivc, n_ivc, i_ivc = build_tube_mesh(ivc_curve, 0.30, radial_segments=20)
        outer_vessels.append((v_ivc, n_ivc, i_ivc, (0.22, 0.38, 0.85, 0.36)))
        v_ivci, n_ivci, i_ivci = build_tube_mesh(ivc_curve, 0.28, radial_segments=16, arc_range=(3.0, 6.2))
        inner_vessels.append((v_ivci, n_ivci, i_ivci, (0.20, 0.32, 0.70)))

        # 4. Four Pulmonary Veins entering Left Atrium
        pv_specs = [
            [[ 1.85, 1.25, -0.45], [ 1.35, 1.05, -0.42], [ 0.92, 0.92, -0.38]],
            [[ 1.80, 0.65, -0.48], [ 1.32, 0.72, -0.42], [ 0.90, 0.75, -0.38]],
            [[-0.35, 1.15, -0.65], [ 0.25, 1.02, -0.55], [ 0.75, 0.92, -0.42]],
            [[-0.30, 0.68, -0.68], [ 0.25, 0.72, -0.56], [ 0.75, 0.74, -0.42]],
        ]
        for pv_pts in pv_specs:
            pvc = catmull_rom_spline(pv_pts, num_samples=18)
            vpv, npv, ipv = build_tube_mesh(pvc, 0.14, radial_segments=14)
            inner_vessels.append((vpv, npv, ipv, (0.82, 0.18, 0.24)))

        return outer_vessels, inner_vessels

    def _build_coronary_vessels(self):
        """
        Builds solid epicardial Coronary Arteries (LAD, Circumflex, RCA) and
        the Great Cardiac Vein coursing along the anatomical sulci.
        """
        coronaries = []

        # 1. Left Anterior Descending (LAD) Artery + Diagonal Branch
        lad_pts = [
            [ 0.28,  0.68,  0.18],
            [ 0.12,  0.38,  0.68],
            [ 0.16, -0.25,  0.86],
            [ 0.34, -0.98,  0.76],
            [ 0.58, -1.62,  0.52],
        ]
        lad_curve = catmull_rom_spline(lad_pts, num_samples=34)
        lad_rad = np.linspace(0.052, 0.022, 34)
        v, n, i = build_tube_mesh(lad_curve, lad_rad, radial_segments=10)
        coronaries.append((v, n, i, (0.92, 0.12, 0.15)))

        diag_pts = [
            [ 0.16, -0.15,  0.85],
            [ 0.62, -0.52,  0.78],
            [ 0.95, -1.02,  0.52],
        ]
        diag_curve = catmull_rom_spline(diag_pts, num_samples=20)
        diag_rad = np.linspace(0.038, 0.018, 20)
        v, n, i = build_tube_mesh(diag_curve, diag_rad, radial_segments=10)
        coronaries.append((v, n, i, (0.92, 0.12, 0.15)))

        # 2. Left Circumflex (LCX) Artery
        lcx_pts = [
            [ 0.28,  0.68,  0.18],
            [ 0.85,  0.42,  0.32],
            [ 1.25,  0.18, -0.15],
            [ 1.05, -0.35, -0.58],
        ]
        lcx_curve = catmull_rom_spline(lcx_pts, num_samples=26)
        lcx_rad = np.linspace(0.046, 0.020, 26)
        v, n, i = build_tube_mesh(lcx_curve, lcx_rad, radial_segments=10)
        coronaries.append((v, n, i, (0.90, 0.12, 0.15)))

        # 3. Right Coronary Artery (RCA)
        rca_pts = [
            [ 0.12,  0.65,  0.22],
            [-0.55,  0.35,  0.62],
            [-1.12, -0.05,  0.38],
            [-1.05, -0.52, -0.25],
            [-0.45, -0.95, -0.58],
        ]
        rca_curve = catmull_rom_spline(rca_pts, num_samples=32)
        rca_rad = np.linspace(0.050, 0.022, 32)
        v, n, i = build_tube_mesh(rca_curve, rca_rad, radial_segments=10)
        coronaries.append((v, n, i, (0.90, 0.12, 0.15)))

        # 4. Great Cardiac Vein (companion blue venous vessel)
        gcv_pts = [
            [ 0.48, -1.52,  0.56],
            [ 0.24, -0.88,  0.80],
            [ 0.06, -0.18,  0.88],
            [ 0.72,  0.35,  0.42],
            [ 1.18,  0.22, -0.22],
        ]
        gcv_curve = catmull_rom_spline(gcv_pts, num_samples=30)
        gcv_rad = np.linspace(0.020, 0.044, 30)
        v, n, i = build_tube_mesh(gcv_curve, gcv_rad, radial_segments=10)
        coronaries.append((v, n, i, (0.18, 0.36, 0.86)))

        return coronaries

    def _build_all_geometry(self):
        self.outer_shell = self._build_outer_epicardial_shell()
        self.septum_mesh = self._build_interventricular_septum()
        self.chamber_cradles = self._build_inner_chamber_cradles()
        self.papillary_meshes = self._build_papillary_muscles()
        self.outer_vessels, self.inner_vessels = self._build_great_vessels()
        self.coronary_meshes = self._build_coronary_vessels()


# =============================================================================
# 4 MULTI-LEAFLET HEART VALVES & DYNAMIC CHORDAE TENDINEAE
# =============================================================================

class HeartValveSystem:
    """
    Renders and animates all 4 cardiac valves in 3D:
      1. Tricuspid Valve (3 cusps, Right AV, opens downward into RV)
      2. Mitral / Bicuspid Valve (2 cusps, Left AV, opens downward into LV)
      3. Pulmonary Valve (3 semilunar cusps, opens upward into Pulmonary Trunk)
      4. Aortic Valve (3 semilunar cusps, opens upward into Ascending Aorta)
    Also returns dynamic 3D attachment coordinates for the Chordae Tendineae.
    """
    def __init__(self):
        self.valves = {
            "tricuspid": {
                "pos": np.array([-0.78,  0.22,  0.18], dtype=np.float32),
                "normal": normalize(np.array([0.12, -0.96, 0.14], dtype=np.float32)),
                "radius": 0.34,
                "num_leaflets": 3,
                "open_dir": 1.0,  # Opens along normal (downward into RV)
                "label": "Tricuspid Valve (3-Cusp)"
            },
            "mitral": {
                "pos": np.array([ 0.72,  0.26, -0.12], dtype=np.float32),
                "normal": normalize(np.array([0.06, -0.96, 0.16], dtype=np.float32)),
                "radius": 0.32,
                "num_leaflets": 2,
                "open_dir": 1.0,  # Opens along normal (downward into LV)
                "label": "Mitral Valve (2-Cusp)"
            },
            "pulmonary": {
                "pos": np.array([-0.25,  0.86,  0.50], dtype=np.float32),
                "normal": normalize(np.array([-0.05, 0.94, -0.22], dtype=np.float32)),
                "radius": 0.25,
                "num_leaflets": 3,
                "open_dir": 1.0,  # Opens along normal (upward into Pulmonary Trunk)
                "label": "Pulmonary Valve (3-Cusp)"
            },
            "aortic": {
                "pos": np.array([ 0.20,  0.60,  0.04], dtype=np.float32),
                "normal": normalize(np.array([0.10, 0.96, 0.16], dtype=np.float32)),
                "radius": 0.26,
                "num_leaflets": 3,
                "open_dir": 1.0,  # Opens along normal (upward into Ascending Aorta)
                "label": "Aortic Valve (3-Cusp)"
            },
        }

    def render_valves_and_chordae(self, cycle: CardiacCycleEngine, papillary_anchors):
        """Renders all 4 solid fibrous rings, animated valve cusps, and chordae tendineae."""
        open_states = {
            "tricuspid": cycle.tricuspid_open,
            "mitral": cycle.mitral_open,
            "pulmonary": cycle.pulmonary_open,
            "aortic": cycle.aortic_open,
        }

        leaflet_tips_world = {"tricuspid": [], "mitral": []}

        # Material for solid fibrous annulus & pearl-white valve leaflets
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.55, 0.55, 0.55, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 48.0)

        for v_name, cfg in self.valves.items():
            # Deform the valve center slightly with the cardiac cycle
            c_def = deform_heart_vertices(cfg["pos"].reshape(1, 3), cycle)[0]
            n_vec = cfg["normal"]
            up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if abs(np.dot(n_vec, up_ref)) > 0.9:
                up_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            u_vec = normalize(np.cross(n_vec, up_ref))
            v_vec = normalize(np.cross(n_vec, u_vec))

            rad = cfg["radius"]
            n_leaf = cfg["num_leaflets"]
            open_amt = open_states[v_name]

            # 1. Render solid Fibrous Annulus Ring (Cardiac Skeleton)
            glColor3f(0.88, 0.82, 0.74)
            n_ring = 28
            r_tube = 0.036
            for i in range(n_ring):
                t0 = (2.0 * math.pi * i) / n_ring
                t1 = (2.0 * math.pi * (i + 1)) / n_ring
                glBegin(GL_QUAD_STRIP)
                for k in range(9):
                    phi = (2.0 * math.pi * k) / 8
                    for t_ang in (t0, t1):
                        radial = math.cos(t_ang) * u_vec + math.sin(t_ang) * v_vec
                        norm_dir = math.cos(phi) * radial + math.sin(phi) * n_vec
                        pt = c_def + (rad + r_tube * math.cos(phi)) * radial + (r_tube * math.sin(phi)) * n_vec
                        glNormal3fv(norm_dir)
                        glVertex3fv(pt)
                glEnd()

            # 2. Render Individual Animated Cusp Leaflets
            # When closed (open_amt = 0), leaflets meet tightly in the center with a slight dome.
            # When open (open_amt = 1), leaflets swing along n_vec and part radially.
            glColor3f(0.96, 0.93, 0.86)
            for l_idx in range(n_leaf):
                ang_start = (2.0 * math.pi * l_idx) / n_leaf + 0.04
                ang_end = (2.0 * math.pi * (l_idx + 1)) / n_leaf - 0.04
                ang_mid = 0.5 * (ang_start + ang_end)

                n_arc = 10
                arcs = np.linspace(ang_start, ang_end, n_arc)

                # Radial tip position depends on valve opening
                tip_radial_dist = rad * (0.04 + 0.82 * open_amt)
                tip_axial_drop = rad * (0.14 + 0.78 * open_amt)

                mid_radial = math.cos(ang_mid) * u_vec + math.sin(ang_mid) * v_vec
                tip_world = c_def + tip_radial_dist * mid_radial + tip_axial_drop * n_vec

                if v_name in leaflet_tips_world:
                    leaflet_tips_world[v_name].append(tip_world)

                # Draw curved triangular/semilunar cusp surface with thickness
                glBegin(GL_TRIANGLE_STRIP)
                for a in arcs:
                    rim_dir = math.cos(a) * u_vec + math.sin(a) * v_vec
                    rim_pt = c_def + rad * rim_dir

                    # Curvature along the leaflet belly
                    angular_center_weight = math.cos((a - ang_mid) * (n_leaf / 2.0))
                    inner_r = rad * (1.0 - (0.96 - 0.80 * open_amt) * angular_center_weight)
                    inner_z = tip_axial_drop * angular_center_weight
                    inner_pt = c_def + inner_r * rim_dir + inner_z * n_vec

                    leaf_norm = normalize(n_vec - 0.55 * rim_dir * open_amt)
                    glNormal3fv(leaf_norm)
                    glVertex3fv(rim_pt)
                    glVertex3fv(inner_pt)
                glEnd()

        # 3. Render Dynamic Solid Chordae Tendineae (Tendinous Cords in RV and LV)
        glColor3f(0.92, 0.87, 0.78)
        glLineWidth(2.2)
        for anchor in papillary_anchors:
            v_type = anchor["valve"]
            l_idx = anchor["leaf_idx"]
            tips = leaflet_tips_world.get(v_type, [])
            if l_idx < len(tips):
                pap_tip = deform_heart_vertices(anchor["tip"].reshape(1, 3), cycle)[0]
                leaf_tip = tips[l_idx]

                # Fan out 3 distinct tendinous cords from each papillary muscle head to the valve cusp
                for offset_scale in (-0.05, 0.0, 0.05):
                    cord_end = leaf_tip + np.array([offset_scale, 0.0, -offset_scale], dtype=np.float32)
                    cord_pts = catmull_rom_spline([
                        pap_tip,
                        0.5 * (pap_tip + cord_end) + np.array([0.0, 0.02, 0.02]),
                        cord_end
                    ], num_samples=8)
                    glBegin(GL_LINE_STRIP)
                    for cp in cord_pts:
                        glNormal3f(0.0, 1.0, 0.0)
                        glVertex3fv(cp)
                    glEnd()


# =============================================================================
# HEMODYNAMIC BLOOD FLOW PARTICLE SYSTEM (VALVE-GATED)
# =============================================================================

class BloodFlowSystem:
    """
    Simulates 380 solid 3D erythrocyte (red blood cell) particles flowing along
    4 anatomical pathways (2 Deoxygenated Right-Heart streams, 2 Oxygenated Left-Heart streams).
    Particles obey the opening/closing of the 4 valves and accelerate with systolic pressure waves.
    """
    def __init__(self, num_particles=380):
        self.num_particles = num_particles
        self._build_hemodynamic_splines()
        self._build_erythrocyte_display_list()
        self.reset_particles()

    def _build_hemodynamic_splines(self):
        # Landmark parameter s along each 200-sample spline:
        #   s ~ 0.36 : AV Valve Gate (Tricuspid / Mitral)
        #   s ~ 0.64 : Semilunar Valve Gate (Pulmonary / Aortic)
        
        # Path 0: Superior Vena Cava -> RA -> Tricuspid -> RV -> Pulmonary Valve -> Left Pulmonary Artery
        p0 = [
            [-1.12,  2.55, -0.15],
            [-1.12,  1.65, -0.08],
            [-1.12,  0.82,  0.02],  # Right Atrium
            [-0.78,  0.22,  0.18],  # Tricuspid Valve (s ~ 0.36)
            [-0.72, -0.58,  0.24],  # Right Ventricle inflow
            [-0.25, -1.25,  0.30],  # RV Apex turn
            [-0.24, -0.15,  0.46],  # Infundibulum
            [-0.25,  0.86,  0.50],  # Pulmonary Valve (s ~ 0.64)
            [-0.24,  1.52,  0.32],  # Pulmonary Trunk
            [ 0.35,  1.70, -0.18],
            [ 1.42,  1.62, -0.40],  # Left Pulmonary Artery
        ]

        # Path 1: Inferior Vena Cava -> RA -> Tricuspid -> RV -> Pulmonary Valve -> Right Pulmonary Artery
        p1 = [
            [-1.12, -1.65, -0.35],
            [-1.12, -0.45, -0.18],
            [-1.08,  0.68,  0.04],  # Right Atrium
            [-0.78,  0.22,  0.18],  # Tricuspid Valve
            [-0.82, -0.65,  0.20],  # Right Ventricle
            [-0.32, -1.32,  0.28],  # RV Apex turn
            [-0.26, -0.10,  0.48],  # Infundibulum
            [-0.25,  0.86,  0.50],  # Pulmonary Valve
            [-0.22,  1.55,  0.25],  # Pulmonary Trunk
            [-0.72,  1.62, -0.12],
            [-1.35,  1.56, -0.28],  # Right Pulmonary Artery
        ]

        # Path 2: Left Pulmonary Veins -> LA -> Mitral -> LV -> Aortic Valve -> Aortic Arch -> Descending Aorta
        p2 = [
            [ 1.78,  1.15, -0.44],
            [ 1.25,  0.96, -0.40],
            [ 0.84,  0.76, -0.32],  # Left Atrium
            [ 0.72,  0.26, -0.12],  # Mitral Valve (s ~ 0.36)
            [ 0.82, -0.58,  0.06],  # Left Ventricle inflow
            [ 0.66, -1.55,  0.24],  # LV Apex turn
            [ 0.36, -0.42,  0.10],  # LV Outflow Tract
            [ 0.20,  0.60,  0.04],  # Aortic Valve (s ~ 0.64)
            [ 0.24,  1.45,  0.12],  # Ascending Aorta
            [-0.12,  2.16, -0.12],  # Aortic Arch
            [-0.46,  1.45, -0.75],
            [-0.38, -1.45, -0.85],  # Descending Aorta
        ]

        # Path 3: Right Pulmonary Veins -> LA -> Mitral -> LV -> Aortic Valve -> Aortic Arch Branches
        p3 = [
            [-0.28,  0.95, -0.65],
            [ 0.32,  0.88, -0.52],
            [ 0.80,  0.72, -0.32],  # Left Atrium
            [ 0.72,  0.26, -0.12],  # Mitral Valve
            [ 0.76, -0.62,  0.02],  # Left Ventricle
            [ 0.60, -1.52,  0.22],  # LV Apex
            [ 0.34, -0.38,  0.08],  # LV Outflow Tract
            [ 0.20,  0.60,  0.04],  # Aortic Valve
            [ 0.22,  1.52,  0.10],  # Ascending Aorta
            [-0.06,  2.14, -0.08],  # Aortic Arch
            [-0.05,  2.85, -0.08],  # Systemic Arch Branch
        ]

        self.splines = [
            catmull_rom_spline(p0, num_samples=220),
            catmull_rom_spline(p1, num_samples=220),
            catmull_rom_spline(p2, num_samples=220),
            catmull_rom_spline(p3, num_samples=220),
        ]
        # Precompute spline tangents for orienting particles and radial offsets
        self.tangents = [normalize(np.gradient(sp, axis=0)) for sp in self.splines]

    def _build_erythrocyte_display_list(self):
        """Compiles a hardware display list for a 3D biconcave red blood cell corpuscle."""
        self.rbc_list = glGenLists(1)
        glNewList(self.rbc_list, GL_COMPILE)
        n_lat, n_lon = 10, 14
        radius = 0.046
        for i in range(n_lat):
            u0 = math.pi * i / n_lat
            u1 = math.pi * (i + 1) / n_lat
            glBegin(GL_QUAD_STRIP)
            for j in range(n_lon + 1):
                v = 2.0 * math.pi * j / n_lon
                for u in (u0, u1):
                    r_xy = math.sin(u)
                    # Biconcave disc profile equation
                    z_prof = 0.42 * math.cos(u) * (0.35 + 0.65 * (r_xy ** 2))
                    x = radius * r_xy * math.cos(v)
                    y = radius * r_xy * math.sin(v)
                    z = radius * z_prof
                    norm = normalize(np.array([x, y, z * 2.2]))
                    glNormal3fv(norm)
                    glVertex3f(x, y, z)
            glEnd()
        glEndList()

    def reset_particles(self):
        rng = np.random.default_rng(42)
        self.path_id = rng.integers(0, 4, size=self.num_particles)
        self.s = rng.uniform(0.0, 1.0, size=self.num_particles).astype(np.float32)
        self.radial_angle = rng.uniform(0.0, 2.0 * math.pi, size=self.num_particles).astype(np.float32)
        self.radial_dist = rng.uniform(0.01, 0.095, size=self.num_particles).astype(np.float32)
        self.speed_mult = rng.uniform(0.86, 1.16, size=self.num_particles).astype(np.float32)
        self.current_vel = np.full(self.num_particles, 0.25, dtype=np.float32)

    def update(self, dt, cycle: CardiacCycleEngine):
        if dt <= 0.0:
            return

        av_gate_s = 0.30
        sl_gate_s = 0.64

        for i in range(self.num_particles):
            pid = self.path_id[i]
            s = self.s[i]
            is_right_heart = pid in (0, 1)

            av_open = cycle.tricuspid_open if is_right_heart else cycle.mitral_open
            sl_open = cycle.pulmonary_open if is_right_heart else cycle.aortic_open

            # Determine physiological velocity along the cardiac pathway
            if s < 0.20:
                # Venous inflow (steady with slight diastolic suction boost)
                vel = 0.28 + 0.14 * av_open
            elif s < av_gate_s:
                # Inside Atrium: surges during Atrial Systole when AV valve is open
                if av_open > 0.15:
                    vel = 0.32 + 0.65 * max(0.0, cycle.atrial_contraction) + 0.22 * av_open
                else:
                    # Swirl/collect inside Atrium while AV valve is closed
                    vel = 0.05
                    self.radial_angle[i] += 4.5 * dt
            elif s < sl_gate_s:
                # Inside Ventricle: fills during diastole, accelerates toward outflow during systole
                if s < 0.48:
                    vel = 0.38 * av_open + 0.22
                else:
                    if sl_open > 0.15:
                        vel = 0.45 + 0.95 * cycle.ventricular_contraction
                    else:
                        vel = 0.06
                        self.radial_angle[i] += 5.0 * dt
            else:
                # Inside Great Arteries (Pulmonary Artery / Aorta): pulsatile systolic wave
                vel = 0.24 + 0.98 * cycle.arterial_pulse

            vel *= self.speed_mult[i]
            s_next = s + vel * dt

            # Hard physical valve gating: prevent crossing a closed valve!
            if s < av_gate_s and s_next >= av_gate_s and av_open < 0.20:
                s_next = av_gate_s - 0.004
                vel = 0.02
            if s < sl_gate_s and s_next >= sl_gate_s and sl_open < 0.20:
                s_next = sl_gate_s - 0.004
                vel = 0.02

            if s_next >= 1.0:
                s_next -= 1.0

            self.s[i] = s_next
            self.current_vel[i] = vel
            self.radial_angle[i] += 1.8 * dt

    def render(self, cycle: CardiacCycleEngine):
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.65, 0.65, 0.65, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 55.0)

        for i in range(self.num_particles):
            pid = self.path_id[i]
            sp = self.splines[pid]
            tg = self.tangents[pid]

            idx_f = self.s[i] * (len(sp) - 1)
            idx0 = int(idx_f)
            idx1 = min(idx0 + 1, len(sp) - 1)
            frac = idx_f - idx0

            base_pt = (1.0 - frac) * sp[idx0] + frac * sp[idx1]
            tangent = normalize((1.0 - frac) * tg[idx0] + frac * tg[idx1])

            # Radial offset inside lumen
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if abs(np.dot(tangent, up)) > 0.9:
                up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            n1 = normalize(np.cross(tangent, up))
            n2 = normalize(np.cross(tangent, n1))

            ang = self.radial_angle[i]
            dist = self.radial_dist[i]
            world_pt = base_pt + dist * (math.cos(ang) * n1 + math.sin(ang) * n2)

            # Deform blood position slightly in sync with chamber walls
            world_pt = deform_heart_vertices(world_pt.reshape(1, 3), cycle)[0]

            # Color: Deoxygenated venous blood (Deep Blue) vs Oxygenated arterial blood (Bright Crimson)
            if pid in (0, 1):
                glColor3f(0.14, 0.44, 0.98)
                trail_col = (0.22, 0.55, 1.0)
            else:
                glColor3f(0.98, 0.12, 0.18)
                trail_col = (1.0, 0.28, 0.32)

            # Draw solid 3D erythrocyte corpuscle
            glPushMatrix()
            glTranslatef(world_pt[0], world_pt[1], world_pt[2])
            glCallList(self.rbc_list)
            glPopMatrix()

            # Draw hydrodynamic velocity streak behind moving blood cells
            streak_len = min(0.16, 0.14 * self.current_vel[i])
            if streak_len > 0.02:
                tail_pt = world_pt - streak_len * tangent
                glDisable(GL_LIGHTING)
                glLineWidth(1.8)
                glBegin(GL_LINES)
                glColor3fv(trail_col)
                glVertex3fv(world_pt)
                glColor3f(trail_col[0] * 0.35, trail_col[1] * 0.35, trail_col[2] * 0.35)
                glVertex3fv(tail_pt)
                glEnd()
                glEnable(GL_LIGHTING)


# =============================================================================
# 2D HUD OVERLAY & INTERACTIVE BUTTON MANAGER
# =============================================================================

class HUDOverlay:
    """Renders crisp 2D text panels, live ECG waveform, valve status bars, and clickable [P] / [R] buttons."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        pygame.font.init()
        self.font_title = pygame.font.SysFont("Consolas", 17, bold=True)
        self.font_body = pygame.font.SysFont("Consolas", 14, bold=False)
        self.font_btn = pygame.font.SysFont("Consolas", 15, bold=True)
        self.tex_cache = {}

        # Define clickable UI buttons for Pause [P] and Reset [R]
        self.btn_pause_rect = pygame.Rect(20, height - 58, 175, 38)
        self.btn_reset_rect = pygame.Rect(210, height - 58, 175, 38)

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

        if len(self.tex_cache) > 150:
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

    def render(self, cycle: CardiacCycleEngine, cam_rot, cam_zoom, paused, mouse_pos):
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
        self.draw_box(16, 16, 435, 252, (0.05, 0.08, 0.14, 0.78), (0.28, 0.45, 0.68, 0.85))
        self.draw_text(28, 26, "3D ANATOMICAL HEART & HEMODYNAMICS", self.font_title, (255, 110, 125))
        self.draw_text(28, 50, f"Cardiac Phase : {cycle.phase_name[:34]}", self.font_body, (230, 240, 255))
        self.draw_text(28, 70, f"Heart Rate    : {cycle.bpm:.0f} BPM  (Cycle: {cycle.phase*100:04.1f}%)", self.font_body, (185, 225, 255))

        # Camera X, Y, Z Angles readout
        rx, ry, rz = cam_rot
        self.draw_text(28, 96, f"Camera Angles : X={rx:6.1f}° | Y={ry:6.1f}° | Z={rz:6.1f}°", self.font_body, (160, 250, 195))
        self.draw_text(28, 118, "Direction Keys Controls (Full 3-Axis Rotation):", self.font_body, (255, 215, 130))
        self.draw_text(28, 138, "  [UP / DOWN]         : Rotate X-Axis (Pitch)", self.font_body, (210, 220, 235))
        self.draw_text(28, 156, "  [LEFT / RIGHT]      : Rotate Y-Axis (Yaw)", self.font_body, (210, 220, 235))
        self.draw_text(28, 174, "  [SHIFT+LEFT / RIGHT]: Rotate Z-Axis (Roll) [or Q/E]", self.font_body, (210, 220, 235))
        self.draw_text(28, 192, "  [W / S] or Scroll   : Zoom Camera In / Out", self.font_body, (210, 220, 235))
        self.draw_text(28, 210, "  [P] Pause/Resume    | [R] Reset Simulation", self.font_body, (210, 220, 235))
        self.draw_text(28, 234, "Legend: [Blue] Deoxygenated (RA/RV) | [Red] Oxygenated (LA/LV)", self.font_body, (195, 210, 245))

        # 2. Right Panel: Live Valve Status & Real-Time ECG Strip
        rx_box = self.width - 340
        self.draw_box(rx_box, 16, 324, 230, (0.05, 0.08, 0.14, 0.78), (0.28, 0.45, 0.68, 0.85))
        self.draw_text(rx_box + 14, 24, "VALVE DYNAMICS & LIVE ECG", self.font_title, (130, 230, 255))

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
            # Progress bar
            bar_x = rx_box + 225
            self.draw_box(bar_x, vy + 3, 82, 12, (0.18, 0.22, 0.28, 0.9))
            fill_col = (0.25, 0.92, 0.48, 0.95) if v_open > 0.25 else (0.92, 0.28, 0.28, 0.85)
            self.draw_box(bar_x, vy + 3, max(3.0, 82.0 * v_open), 12, fill_col)

        # Live ECG Waveform Plot
        ecg_x, ecg_y, ecg_w, ecg_h = rx_box + 14, 158, 296, 74
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

        # 3. Interactive On-Screen Buttons: [P] PAUSE and [R] RESET
        self.btn_pause_rect.y = self.height - 56
        self.btn_reset_rect.y = self.height - 56

        p_hover = self.btn_pause_rect.collidepoint(mouse_pos)
        r_hover = self.btn_reset_rect.collidepoint(mouse_pos)

        p_fill = (0.85, 0.48, 0.14, 0.92) if paused else ((0.18, 0.36, 0.58, 0.92) if p_hover else (0.10, 0.22, 0.38, 0.86))
        r_fill = (0.62, 0.18, 0.24, 0.92) if r_hover else (0.10, 0.22, 0.38, 0.86)

        self.draw_box(
            self.btn_pause_rect.x, self.btn_pause_rect.y,
            self.btn_pause_rect.w, self.btn_pause_rect.h,
            p_fill, (0.55, 0.78, 1.0, 0.95)
        )
        p_label = "[P] RESUME" if paused else "[P] PAUSE"
        self.draw_text(self.btn_pause_rect.x + 36, self.btn_pause_rect.y + 10, p_label, self.font_btn, (255, 255, 255))

        self.draw_box(
            self.btn_reset_rect.x, self.btn_reset_rect.y,
            self.btn_reset_rect.w, self.btn_reset_rect.h,
            r_fill, (0.55, 0.78, 1.0, 0.95)
        )
        self.draw_text(self.btn_reset_rect.x + 38, self.btn_reset_rect.y + 10, "[R] RESET", self.font_btn, (255, 255, 255))

        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)


# =============================================================================
# OPENGL MESH RENDERING HELPER
# =============================================================================

def draw_deformed_mesh(base_verts, indices, cycle: CardiacCycleEngine, color_rgba, shininess=36.0):
    """Deforms a 3D mesh according to the current cardiac cycle state and renders it via Vertex Arrays."""
    def_verts = deform_heart_vertices(base_verts, cycle)
    def_norms = compute_vertex_normals(def_verts, indices)

    if len(color_rgba) == 3:
        glColor3fv(color_rgba)
    else:
        glColor4fv(color_rgba)

    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, def_verts)
    glNormalPointer(GL_FLOAT, 0, def_norms)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, indices)
    glDisableClientState(GL_NORMAL_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)


# =============================================================================
# MAIN APPLICATION LOOP
# =============================================================================

def main():
    pygame.init()
    width, height = 1280, 720
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("PyOpenGL 3D Beating Heart Simulation - Valves, Hemodynamics & Translucent Shell")

    # Configure OpenGL state & 3-Point Studio Medical Lighting
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glEnable(GL_NORMALIZE)
    glShadeModel(GL_SMOOTH)
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)

    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    # Key Light (Warm Anterior-Right)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [4.5, 5.0, 6.5, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.95, 0.92, 0.90, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.90, 0.90, 0.90, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.22, 0.20, 0.24, 1.0])

    # Rim / Fill Light (Cool Posterior-Left)
    glEnable(GL_LIGHT1)
    glLightfv(GL_LIGHT1, GL_POSITION, [-5.5, 2.5, -4.5, 1.0])
    glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.38, 0.46, 0.65, 1.0])
    glLightfv(GL_LIGHT1, GL_SPECULAR, [0.35, 0.40, 0.55, 1.0])

    # Inferior Apex Fill Light
    glEnable(GL_LIGHT2)
    glLightfv(GL_LIGHT2, GL_POSITION, [0.0, -6.0, 3.0, 1.0])
    glLightfv(GL_LIGHT2, GL_DIFFUSE, [0.28, 0.22, 0.25, 1.0])

    # Projection Matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(42.0, width / float(height), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

    # Instantiate Anatomical & Physiological Systems
    cycle = CardiacCycleEngine(bpm=72.0)
    heart = AnatomicalHeartModel()
    valves = HeartValveSystem()
    blood = BloodFlowSystem(num_particles=380)
    hud = HUDOverlay(width, height)

    # Camera State (X, Y, Z Euler rotations in degrees + zoom distance)
    init_cam_rot = [8.0, -14.0, 0.0]
    cam_rot = list(init_cam_rot)
    cam_zoom = 7.4
    paused = False

    def reset_simulation():
        nonlocal cam_rot, cam_zoom, paused
        cam_rot = list(init_cam_rot)
        cam_zoom = 7.4
        paused = False
        cycle.reset()
        blood.reset_particles()

    clock = pygame.time.Clock()
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
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
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    if hud.btn_pause_rect.collidepoint(event.pos):
                        paused = not paused
                    elif hud.btn_reset_rect.collidepoint(event.pos):
                        reset_simulation()
                elif event.button == 4:
                    cam_zoom = max(3.5, cam_zoom - 0.35)
                elif event.button == 5:
                    cam_zoom = min(14.0, cam_zoom + 0.35)

        # Continuous Keyboard Direction Keys for X, Y, Z Camera Rotation
        keys = pygame.key.get_pressed()
        rot_speed = 75.0 * dt
        shift_held = keys[K_LSHIFT] or keys[K_RSHIFT]

        # X-Axis rotation (UP / DOWN direction keys)
        if keys[K_UP]:
            cam_rot[0] = (cam_rot[0] - rot_speed) % 360.0
        if keys[K_DOWN]:
            cam_rot[0] = (cam_rot[0] + rot_speed) % 360.0

        # Y-Axis & Z-Axis rotation via LEFT / RIGHT direction keys (hold SHIFT for Z-Axis Roll, or use Q/E / PageUp/PageDown)
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

        # Dedicated keys for Z-axis rotation as well
        if keys[K_q] or keys[K_PAGEUP]:
            cam_rot[2] = (cam_rot[2] + rot_speed) % 360.0
        if keys[K_e] or keys[K_PAGEDOWN]:
            cam_rot[2] = (cam_rot[2] - rot_speed) % 360.0

        # Zoom controls
        if keys[K_w] or keys[K_PLUS] or keys[K_EQUALS]:
            cam_zoom = max(3.5, cam_zoom - 3.5 * dt)
        if keys[K_s] or keys[K_MINUS]:
            cam_zoom = min(14.0, cam_zoom + 3.5 * dt)

        # Update Physiological Cardiac Cycle & Blood Flow
        sim_dt = 0.0 if paused else dt
        cycle.update(sim_dt)
        blood.update(sim_dt, cycle)

        # Clear Framebuffer
        glClearColor(0.035, 0.045, 0.075, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        glTranslatef(0.0, -0.15, -cam_zoom)
        glRotatef(cam_rot[0], 1.0, 0.0, 0.0)  # X-Axis Pitch
        glRotatef(cam_rot[1], 0.0, 1.0, 0.0)  # Y-Axis Yaw
        glRotatef(cam_rot[2], 0.0, 0.0, 1.0)  # Z-Axis Roll

        # =====================================================================
        # PASS 1: OPAQUE SOLID INNER ANATOMY (Septum, Chambers, Valves, Blood)
        # =====================================================================
        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.42, 0.35, 0.35, 1.0])

        # 1A. Solid Muscular Interventricular & Interatrial Septum
        vs, _, idx_s = heart.septum_mesh
        draw_deformed_mesh(vs, idx_s, cycle, (0.64, 0.14, 0.18), shininess=28.0)

        # 1B. Solid Sculpted Endocardial Chamber Cradles & Trabeculae Carneae
        for vc, _, idx_c, col_c in heart.chamber_cradles:
            draw_deformed_mesh(vc, idx_c, cycle, col_c, shininess=32.0)

        # 1C. Solid Conical Papillary Muscles
        for vp, _, idx_p, col_p in heart.papillary_meshes:
            draw_deformed_mesh(vp, idx_p, cycle, col_p, shininess=38.0)

        # 1D. Solid Inner Great Vessel Lumina & Branches
        for vv, _, idx_v, col_v in heart.inner_vessels:
            draw_deformed_mesh(vv, idx_v, cycle, col_v, shininess=40.0)

        # 1E. Solid Epicardial Coronary Arteries & Cardiac Veins
        for vcor, _, idx_cor, col_cor in heart.coronary_meshes:
            draw_deformed_mesh(vcor, idx_cor, cycle, col_cor, shininess=55.0)

        # 1F. Solid 4 Cardiac Valves (Tricuspid, Mitral, Pulmonary, Aortic) & Chordae Tendineae
        valves.render_valves_and_chordae(cycle, heart.papillary_anchors)

        # 1G. Solid 3D Erythrocyte Blood Flow Particles
        blood.render(cycle)

        # =====================================================================
        # PASS 2: SEMI-TRANSPARENT OUTER LAYER (Two-Pass Back/Front Culling)
        # =====================================================================
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        glEnable(GL_CULL_FACE)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.85, 0.75, 0.78, 1.0])

        vo_shell, _, idx_shell = heart.outer_shell
        shell_rgba = (0.88, 0.22, 0.28, 0.30)

        # First render back faces, then front faces for artifact-free 3D translucency
        for cull_mode in (GL_FRONT, GL_BACK):
            glCullFace(cull_mode)
            draw_deformed_mesh(vo_shell, idx_shell, cycle, shell_rgba, shininess=72.0)
            for v_out, _, idx_out, col_rgba in heart.outer_vessels:
                draw_deformed_mesh(v_out, idx_out, cycle, col_rgba, shininess=64.0)

        glDisable(GL_CULL_FACE)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

        # =====================================================================
        # PASS 3: 2D HUD TELEMETRY, LIVE ECG & CLICKABLE [P]/[R] BUTTONS
        # =====================================================================
        hud.render(cycle, cam_rot, cam_zoom, paused, mouse_pos)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()