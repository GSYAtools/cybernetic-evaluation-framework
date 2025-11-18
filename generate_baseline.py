import os
import json
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv

from sampler import generate_if_missing
from embedder import embed_outputs
from divergence import compute_divergence

# Cargar API key desde .env para que funcione sampler
load_dotenv()

# Configuración
BASELINE_DIR = "baseline_prompts"
OUTPUT_DIR = "outputs_t2"
OUTFILE = "baseline_thresholds_t2.json"
BINS = 30

def build_baseline():
    prompt_files = [f for f in os.listdir(BASELINE_DIR) if f.endswith(".json")]
    if not prompt_files:
        raise ValueError("No baseline prompt files found in 'baseline_prompts/'")

    all_scores = defaultdict(list)  # métricas → lista de valores

    for fname in prompt_files:
        path = os.path.join(BASELINE_DIR, fname)
        with open(path, encoding="utf-8") as f:
            config = json.load(f)

        prompt_name = config["name"]
        print(f"\n==> Procesando baseline: {prompt_name}")

        # Generar muestras si no existen
        generate_if_missing(config, output_dir=OUTPUT_DIR)

        # Embedir salidas
        A_emb, B_emb = embed_outputs(prompt_name, config, output_dir=OUTPUT_DIR)

        # Calcular divergencias sin bootstrap
        res = compute_divergence(A_emb, B_emb, metrics=config["metrics"], bins=BINS)

        # Acumular resultados
        for metric, value in res.items():
            all_scores[metric].append(value)

    # Calcular umbrales empíricos
    thresholds = {}
    for metric, scores in all_scores.items():
        scores = np.array(scores)
        mu = np.mean(scores)
        sigma = np.std(scores)
        p95 = np.percentile(scores, 95)

        thresholds[metric] = {
            "mean_plus_2sigma": round(float(mu + 2 * sigma), 4),
            "percentile_95": round(float(p95), 4)
        }

    # Guardar umbrales en archivo
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\nUmbrales guardados en {OUTFILE}")

if __name__ == "__main__":
    build_baseline()
