# Complete-Cardiopulmonary-Simulator
Interactive 3D cardiopulmonary simulator built with Python, PyOpenGL, Pygame, and NumPy. Simulates real-time cardiac cycles, 4 animated heart valves, closed-loop alveolar gas exchange, respiratory lung mechanics, procedural S1/S2 stethoscope audio, and view-dependent 360° transparency.


# 3D Cardiopulmonary Simulator

An interactive, real-time 3D simulation of the human heart and lungs built with **Python**, **PyOpenGL**, **Pygame**, and **NumPy**.

## Key Features
- **Synchronized Cardiopulmonary Mechanics:** Couples a 72 BPM cardiac cycle with an 18 breaths/min respiratory cycle, modeling atrial/ventricular contraction, apical torsion, diaphragmatic descent, and alveolar expansion.
- **4 Animated Heart Valves & Papillary Apparatus:** Real-time opening and closing of the **Tricuspid**, **Mitral**, **Pulmonary**, and **Aortic** valves complete with fibrous annuli, papillary muscles, and dynamic chordae tendineae.
- **Closed-Loop Hemodynamics & Alveolar Gas Exchange:** Simulates 520 3D erythrocytes transitioning from deoxygenated venous blue to oxygenated arterial crimson inside the alveolar capillary beds, alongside 220 respiratory $O_2 / CO_2$ airflow particles.
- **View-Dependent 360° Transparency:** Dynamically modulates front-facing and back-facing alpha values based on the 3D camera view vector so whichever side faces the camera acts as a clear window into the solid internal anatomy.
- **Procedural Stethoscope Audio & Live Telemetry:** Synthesizes S1 (*"Lub"*), S2 (*"Dub"*), systolic ejection rush, and vesicular breath sounds in real time alongside live **Lead II ECG**, **Phonocardiogram (PCG)**, and **Spirometry** plots.

## Installation & Usage

```bash
pip install -r requirements.txt
python main.py

