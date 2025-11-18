# Cybernetic Evaluation Framework

This repository provides the implementation and replication materials for the article **"Distributional Behavioural Measurement for Black-Box Systems: A Cybernetic Framework for Observable Evaluation"**.
It implements a modular pipeline to evaluate black-box models from observable input–output behaviour only: (i) observable projection (ψ) via semantic embedding, (ii) distributional comparison using Jensen–Shannon, Wasserstein-1 and Total Variation, and (iii) calibrated interpretation (θ_D) and structural inference (NAP, FIM, LBIP).

The codebase includes scripts to reproduce the paper’s experiments (baseline calibration, temporal drift analysis, bootstrap sensitivity), the data fixtures used in the manuscript, and utilities to export JSON/LaTeX tables and visualisations. The model under test is treated as a black box (the experiments use `gpt-4o-mini` under stochastic sampling); exact evaluation parameters and dependency versions are provided in `requirements.txt` and `environment.md` to facilitate replication.

---
## Components

### Core Modules

| Module                   | Description |
|---------------------------|-------------|
| `prepare_samples.py`      | Performs the initial sampling of all calibration and test prompts used for behavioural measurement under observable projections. |
| `sampler.py`              | Generates observable outputs from the `gpt-4o-mini` model (treated as a black-box system) using stochastic sampling parameters. |
| `embedder.py`             | Projects the generated outputs into a measurable observable space (ψ) through semantic encoding. |
| `divergence.py`           | Computes distributional measures of behavioural difference (Jensen–Shannon divergence, Wasserstein-1 distance, Total Variation distance). |
| `generate_baseline.py`    | Estimates intrinsic variability thresholds (θ<sub>D</sub>) from control prompt pairs via bootstrap and percentile-based calibration. |
| `drift_test.py`           | Implements the **Temporal Drift Validation**, comparing calibrated observable distributions between temporal checkpoints T₁ and T₂, normalising divergences (R<sub>D</sub>), and exporting JSON / LaTeX summaries. |
| `run_case.py`             | Executes complete experimental cases (sampling → embedding → divergence → calibration → report generation), ensuring reproducible pipeline execution. |


### Additional Modules

| Module                | Description |
|-----------------------|-------------|
| `analyze_output.py`   | Extracts representative and divergent output pairs for interpretability and qualitative inspection of observable behavioural regimes. |
| `visualize.py`        | Produces t-SNE, UMAP and kernel density visualisations of observable distributions, supporting analysis of behavioural proximity and activation regions. |
| `sens_anal.py`        | Performs bootstrap-based sensitivity analysis of distributional measures and evaluates the stability of calibrated indicators (θ<sub>D</sub>, R<sub>D</sub>). |
| `report.py`           | Summarises numerical and graphical results for each experimental case, generating reproducible tables and figures aligned with the article’s validation sections. |
| `summary.py`          | Aggregates results from multiple experimental runs, providing consolidated statistics across calibration and drift analyses. |


---

## Step-by-Step Execution

### 1. Build Baseline (once)

```bash
python generate_baseline.py
```

Establishes the empirical reference thresholds (θ<sub>D</sub>) for each behavioural measure under nominal (stable) conditions.  
This step calibrates the system’s intrinsic variability and must be executed once before any comparative experiment.

**Notes**
- Default configuration uses n = 30 samples per prompt (as in the manuscript) and bootstrap resampling for percentile estimation.  
- Reproducibility: see `requirements.txt` and `environment.md` for dependency versions; use `--seed` where available to fix randomness.

---

### 2. Generate Observable Outputs for Calibration or Test Prompts

```bash
python prepare_samples.py --case photosynthesis_gravity --n 30 --seed 42 --temperature 0.7 --top_p 0.9
```

Generates model responses (observable outputs) from the black-box model (`gpt-4o-mini`) for all defined prompt pairs.  
Outputs are stored as raw text under `outputs/<case_name>/` for subsequent embedding and divergence computation.

**Notes**
- If you do not have API access, the repository includes example fixtures under `outputs/` to reproduce downstream steps.  
- Available flags (examples): `--n` (samples per prompt), `--seed`, `--temperature`, `--top_p`, `--outdir`.

---

### 3. Run a Full Experimental Case

```bash
python run_case.py photosynthesis_gravity --outdir results/
```

Executes the complete behavioural evaluation workflow (embedding → divergence → calibration-aware normalisation → report export).  
You can replace `photosynthesis_gravity` with any valid prompt name defined in the `prompts/` folder (omit the `.json` extension).

**Outputs**
- `outputs/<case_name>/A_outputs.json`, `B_outputs.json` (raw outputs)  
- `outputs/<case_name>/A_emb.npy`, `B_emb.npy` (embeddings)  
- `results/<case_name>/report.json`, `results/<case_name>/examples.json`  
- visualisations under `results/<case_name>/`

---

### 4. Local Output Analysis (optional)

```bash
python analyze_output.py photosynthesis_gravity --outdir results/
```

Performs qualitative inspection of observable differences and extracts representative / divergent samples for interpretability and manual review.

---

## Special Experimental Analyses

### Temporal Drift Validation 

```bash
python drift_test.py --t1 outputs/ --t2 outputs_t2/ --thresholds baseline_thresholds.json
```

Compares calibrated behavioural distributions between two temporal checkpoints (T₁ and T₂) to assess system homeostasis and behavioural stability.

**Outputs include**
- Normalised divergence ratios (R<sub>JS</sub>, R<sub>W₁</sub>)  
- JSON summary with mean activation ratios  
- LaTeX table for direct inclusion in reports

**Notes**
- By default `drift_test.py` looks for `OUTPUTS` and `OUTPUTS_T2` directories; use `--t1`/`--t2` to override.

---

### Cross-Observable Inference (LBIP Validation)

No separate execution script is required.  
This analysis is derived from previously computed calibration and drift outputs: it inspects the normalised metrics R<sub>D</sub> exported from prior runs and evaluates proportional bounds consistent with the Lipschitz-Bounded Inference Principle (LBIP).

---

### Bootstrap Sensitivity Analysis

```bash
python sens_anal.py --case photosynthesis_gravity --n_boot 1000
```

Assesses the robustness of the distributional measures under resampling and quantifies the stability of calibration thresholds.

**Outputs**
- Confidence intervals for divergence metrics  
- Variability classification  
- Stability visualisations and empirical convergence plots

---

**Preconditions and reproducibility**
- Provide OpenAI API key via environment variable (if applicable): `export OPENAI_API_KEY=...`.  
- To reproduce without API access, use the included fixtures in `outputs/`.  
- Check `requirements.txt` / `environment.md` for exact package versions and run `pip install -r requirements.txt`.


---


## Expected Outputs

```bash
outputs/<case_name>/A_outputs.json
outputs/<case_name>/B_outputs.json
outputs/<case_name>/A_emb.npy
outputs/<case_name>/B_emb.npy
results/<case_name>/report.json
results/<case_name>/examples.json
results/<case_name>/kde_distances.png
results/<case_name>/tsne_projection.png
results/<case_name>/umap_projection.png
bootstrap_results/<case_name>_JS_bootstrap.png
bootstrap_results/<case_name>_Wasserstein_bootstrap.png
bootstrap_results/bootstrap_summary.json
```

These outputs correspond to a single experimental case (e.g. `photosynthesis_gravity`).  
When multiple calibration or test cases are executed, equivalent files will be generated for each configured scenario defined in the `"prompts/"` folder.

Each result directory includes:

- **Observable outputs (`A_outputs.json`, `B_outputs.json`)**: raw model responses under each input condition.  
- **Embeddings (`A_emb.npy`, `B_emb.npy`)**: vector representations used for behavioural comparison.  
- **Reports (`report.json`)**: calibrated divergence values and normalised ratios (R<sub>D</sub>).  
- **Examples (`examples.json`)**: representative and divergent samples for qualitative inspection.  
- **Visualisations**: kernel density estimates (`kde_distances.png`) and low-dimensional projections (`t-SNE`, `UMAP`) illustrating observable distributions.  
- **Bootstrap results**: variability and stability diagnostics for each behavioural measure, including summary statistics and confidence plots.

---



## Contact

For questions, collaborations or academic inquiries:

**Carlos Mario Braga**  
[carlosmario.braga1@alu.uclm.es](mailto:carlosmario.braga1@alu.uclm.es)



