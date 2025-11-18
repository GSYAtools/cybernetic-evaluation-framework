"""
Temporal Drift Validation (Case B)
----------------------------------
Compares the reference embeddings collected at two temporal checkpoints (T₁ and T₂)
for each calibration prompt pair. For every prompt A and B, it computes JS and
Wasserstein-1 divergences between T₁ and T₂, normalises them using baseline
thresholds, and exports both a JSON and a LaTeX table.
"""

import os
import json
import numpy as np
from datetime import datetime
from divergence import compute_divergence

# --- Configuración ---
OUTPUT_T1 = "OUTPUTS"        # Primer checkpoint (T₁)
OUTPUT_T2 = "OUTPUTS_T2"     # Segundo checkpoint (T₂)
THRESHOLDS_FILE = "baseline_thresholds.json"
METRICS = ["JS", "Wasserstein"]
BINS = 30

# --- Cargar umbrales ---
with open(THRESHOLDS_FILE, encoding="utf-8") as f:
    thresholds = json.load(f)

theta_JS = thresholds.get("JS", {}).get("percentile_95", None)
theta_W1 = thresholds.get("Wasserstein", {}).get("percentile_95", None)

# --- Detectar carpetas comunes ---
folders_T1 = [d for d in os.listdir(OUTPUT_T1) if os.path.isdir(os.path.join(OUTPUT_T1, d))]
folders_T2 = [d for d in os.listdir(OUTPUT_T2) if os.path.isdir(os.path.join(OUTPUT_T2, d))]
common = sorted(list(set(folders_T1).intersection(folders_T2)))

print("Prompts comunes detectados:", common)

# --- Cálculo ---
results = {}
r_js, r_w1 = [], []

for name in common:
    folder1 = os.path.join(OUTPUT_T1, name)
    folder2 = os.path.join(OUTPUT_T2, name)
    print(f"\nProcesando par de calibración: {name}")

    # Cargar embeddings para el mismo prompt en T₁ y T₂
    A_T1 = np.load(os.path.join(folder1, "A_emb.npy"))
    A_T2 = np.load(os.path.join(folder2, "A_emb.npy"))
    B_T1 = np.load(os.path.join(folder1, "B_emb.npy"))
    B_T2 = np.load(os.path.join(folder2, "B_emb.npy"))

    # Divergencia temporal (T₁→T₂) para cada prompt
    div_A = compute_divergence(A_T1, A_T2, metrics=METRICS, bins=BINS)
    div_B = compute_divergence(B_T1, B_T2, metrics=METRICS, bins=BINS)

    # Normalización por umbrales
    R_A_JS = div_A["JS"] / theta_JS if theta_JS else np.nan
    R_A_W1 = div_A["Wasserstein"] / theta_W1 if theta_W1 else np.nan
    R_B_JS = div_B["JS"] / theta_JS if theta_JS else np.nan
    R_B_W1 = div_B["Wasserstein"] / theta_W1 if theta_W1 else np.nan

    results[name] = {
        "promptA": {"JS": div_A["JS"], "W1": div_A["Wasserstein"], "R_JS": R_A_JS, "R_W1": R_A_W1},
        "promptB": {"JS": div_B["JS"], "W1": div_B["Wasserstein"], "R_JS": R_B_JS, "R_W1": R_B_W1}
    }

    r_js.extend([R_A_JS, R_B_JS])
    r_w1.extend([R_A_W1, R_B_W1])

# --- Agregación global ---
mean_R_JS = np.mean(r_js)
mean_R_W1 = np.mean(r_w1)

summary = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "theta_JS": theta_JS,
    "theta_W1": theta_W1,
    "by_prompt": results,
    "mean_R_JS": round(mean_R_JS, 4),
    "mean_R_W1": round(mean_R_W1, 4)
}

# --- Guardar resultados en JSON ---
json_file = f"temporal_drift_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n[FINALIZADO]")
print("Resultados guardados en:", json_file)
print("Media R_JS:", round(mean_R_JS, 3), "  Media R_W1:", round(mean_R_W1, 3))

# --- Exportar tabla LaTeX ---
tex_lines = [
    "\\begin{table}[!t]",
    "\\centering",
    "\\caption{Temporal drift between checkpoints T$_1$ and T$_2$ for calibration prompts}",
    "\\begin{tabular}{lcccc}",
    "\\hline",
    "\\textbf{Prompt pair} & $R_{JS}^{(A)}$ & $R_{JS}^{(B)}$ & $R_{W1}^{(A)}$ & $R_{W1}^{(B)}$ \\\\",
    "\\hline"
]

for name, vals in results.items():
    tex_lines.append(
        f"{name.replace('_',' ')} & "
        f"{vals['promptA']['R_JS']:.3f} & {vals['promptB']['R_JS']:.3f} & "
        f"{vals['promptA']['R_W1']:.3f} & {vals['promptB']['R_W1']:.3f} \\\\"
    )

tex_lines.extend([
    "\\hline",
    f"\\textbf{{Mean}} & \\textbf{{{mean_R_JS:.3f}}} &  & \\textbf{{{mean_R_W1:.3f}}} &  \\\\",
    "\\hline",
    "\\end{tabular}",
    "\\label{tab:temporal_drift}",
    "\\end{table}"
])

tex_file = json_file.replace(".json", ".tex")
with open(tex_file, "w", encoding="utf-8") as f:
    f.write("\n".join(tex_lines))

print("Tabla LaTeX guardada en:", tex_file)
