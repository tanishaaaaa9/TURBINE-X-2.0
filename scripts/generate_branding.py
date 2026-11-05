"""Generate high-resolution TURBINE-X 2.0 branding assets (SVG, PNG logo, PNG icon)."""
import os
import math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

PROJECT_ROOT = Path("D:/TURBINE_X_Full_Project/TURBINE_X_Full_Project")
brand_dir = PROJECT_ROOT / "assets" / "branding"
brand_dir.mkdir(parents=True, exist_ok=True)

# 1. Generate SVG Logo
svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 240" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#070F1E"/>
      <stop offset="100%" stop-color="#0B192C"/>
    </linearGradient>
    <linearGradient id="cyanGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00FFFF"/>
      <stop offset="100%" stop-color="#0096C7"/>
    </linearGradient>
    <linearGradient id="bladeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#00FFFF" stop-opacity="0.9"/>
      <stop offset="100%" stop-color="#0077B6" stop-opacity="0.4"/>
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="4" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>

  <!-- Background rounded card -->
  <rect width="800" height="240" rx="20" fill="url(#bgGrad)" stroke="#1E3E62" stroke-width="2"/>

  <!-- Turbine Rotor Emblem Group -->
  <g transform="translate(130, 120)">
    <!-- Outer energy ring -->
    <circle r="72" fill="none" stroke="#00FFFF" stroke-width="2.5" stroke-dasharray="8 6" opacity="0.6"/>
    <circle r="82" fill="none" stroke="#0077B6" stroke-width="1.5" stroke-dasharray="16 10" opacity="0.4"/>
    
    <!-- 8 Turbine Blades -->
    <!-- Blade 0 -->
    <path d="M 0,-24 C 18,-35 28,-58 20,-70 C 14,-72 4,-62 -8,-45 Z" fill="url(#cyanGrad)" filter="url(#glow)"/>
    <!-- Blade 45 -->
    <g transform="rotate(45)"><path d="M 0,-24 C 18,-35 28,-58 20,-70 C 14,-72 4,-62 -8,-45 Z" fill="url(#cyanGrad)"/></g>
    <!-- Blade 90 -->
    <g transform="rotate(90)"><path d="M 0,-24 C 18,-35 28,-58 20,-70 C 14,-72 4,-62 -8,-45 Z" fill="url(#cyanGrad)"/></g>
    <!-- Blade 135 -->
    <g transform="rotate(135)"><path d="M 0,-24 C 18,-35 28,-58 20,-70 C 14,-72 4,-62 -8,-45 Z" fill="url(#cyanGrad)"/></g>
    <!-- Blade 180 -->
    <g transform="rotate(180)"><path d="M 0,-24 C 18,-35 28,-58 20,-70 C 14,-72 4,-62 -8,-45 Z" fill="url(#cyanGrad)"/></g>
    <!-- Blade 225 -->
    <g transform="rotate(225)"><path d="M 0,-24 C 18,-35 28,-58 20,-70 C 14,-72 4,-62 -8,-45 Z" fill="url(#cyanGrad)"/></g>
    <!-- Blade 270 -->
    <g transform="rotate(270)"><path d="M 0,-24 C 18,-35 28,-58 20,-70 C 14,-72 4,-62 -8,-45 Z" fill="url(#cyanGrad)"/></g>
    <!-- Blade 315 -->
    <g transform="rotate(315)"><path d="M 0,-24 C 18,-35 28,-58 20,-70 C 14,-72 4,-62 -8,-45 Z" fill="url(#cyanGrad)"/></g>

    <!-- Center Hub -->
    <circle r="24" fill="#070F1E" stroke="#00FFFF" stroke-width="3"/>
    <circle r="12" fill="#00FFFF" opacity="0.8"/>
    <!-- Center 'X' Core -->
    <path d="M -6,-6 L 6,6 M 6,-6 L -6,6" stroke="#070F1E" stroke-width="2.5" stroke-linecap="round"/>
  </g>

  <!-- Brand Typography -->
  <text x="250" y="112" font-family="'Segoe UI', 'Roboto', 'Arial', sans-serif" font-weight="900" font-size="58" fill="#FFFFFF" letter-spacing="2">
    TURBINE<tspan fill="#00FFFF">-X</tspan>
  </text>
  <rect x="250" y="128" width="50" height="4" fill="#00FFFF" rx="2"/>
  <text x="312" y="133" font-family="'Segoe UI', 'Roboto', 'Arial', sans-serif" font-weight="700" font-size="16" fill="#00FFFF" letter-spacing="4">
    VERSION 2.0
  </text>
  <text x="250" y="165" font-family="'Segoe UI', 'Roboto', 'Arial', sans-serif" font-weight="400" font-size="16" fill="#90E0EF" letter-spacing="1.2">
    ADVANCED GAS TURBINE AI &amp; MULTI-TASK INTELLIGENCE
  </text>
  <text x="250" y="188" font-family="'Segoe UI', 'Roboto', 'Arial', sans-serif" font-weight="300" font-size="13" fill="#6C757D" letter-spacing="0.8">
    Emission Prediction • Risk Classification • Anomaly Detection • Forecasting
  </text>
</svg>
"""

with open(brand_dir / "turbine_x_logo.svg", "w", encoding="utf-8") as f:
    f.write(svg_content)

# 2. Render High-Resolution PNG Logo and Icon via Matplotlib
fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
fig.patch.set_facecolor('#0B192C')
ax.set_facecolor('#0B192C')
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis('off')

# Draw Rotor
center_x, center_y = 1.6, 1.5
for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
    dx = 0.7 * np.cos(angle)
    dy = 0.7 * np.sin(angle)
    ax.annotate("", xy=(center_x + dx, center_y + dy), xytext=(center_x, center_y),
                arrowprops=dict(arrowstyle="->", color="#00FFFF", lw=3.5, mutation_scale=15))

# Outer ring & hub
circle_outer = plt.Circle((center_x, center_y), 0.85, color='#0096C7', fill=False, lw=1.5, linestyle='--')
circle_hub = plt.Circle((center_x, center_y), 0.3, color='#070F1E', ec='#00FFFF', lw=2.5)
circle_center = plt.Circle((center_x, center_y), 0.15, color='#00FFFF')
ax.add_patch(circle_outer)
ax.add_patch(circle_hub)
ax.add_patch(circle_center)

# Text
ax.text(3.0, 1.7, "TURBINE-X", fontsize=28, fontweight='heavy', color='white', family='sans-serif')
ax.text(6.8, 1.7, "2.0", fontsize=28, fontweight='heavy', color='#00FFFF', family='sans-serif')
ax.plot([3.0, 7.5], [1.45, 1.45], color='#00FFFF', lw=2)
ax.text(3.0, 1.05, "ADVANCED GAS TURBINE AI & MULTI-TASK INTELLIGENCE", fontsize=10, fontweight='bold', color='#90E0EF', family='sans-serif')
ax.text(3.0, 0.7, "Multi-Target Regression • Risk Classification • Anomaly Detection • Forecasting", fontsize=8, color='#6C757D', family='sans-serif')

plt.tight_layout()
plt.savefig(brand_dir / "turbine_x_logo.png", dpi=300, facecolor='#0B192C', edgecolor='none')
plt.close()

# 3. Render Square App Icon (PNG)
fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
fig.patch.set_facecolor('#0B192C')
ax.set_facecolor('#0B192C')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axis('off')

for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
    dx = 1.0 * np.cos(angle)
    dy = 1.0 * np.sin(angle)
    ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#00FFFF", lw=5, mutation_scale=20))

circle_outer = plt.Circle((0, 0), 1.25, color='#0077B6', fill=False, lw=2, linestyle='--')
circle_hub = plt.Circle((0, 0), 0.45, color='#070F1E', ec='#00FFFF', lw=4)
circle_center = plt.Circle((0, 0), 0.22, color='#00FFFF')
ax.add_patch(circle_outer)
ax.add_patch(circle_hub)
ax.add_patch(circle_center)

plt.tight_layout()
plt.savefig(brand_dir / "turbine_x_icon.png", dpi=300, facecolor='#0B192C', edgecolor='none')
plt.close()

print("Branding assets generated: turbine_x_logo.svg, turbine_x_logo.png, turbine_x_icon.png")

