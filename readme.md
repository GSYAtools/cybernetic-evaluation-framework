# Cybernetic Evaluation Framework

This project implements a modular framework for the cybernetic evaluation of black-box systems through observable behaviour. It operationalises the methodology introduced in the article **"Distributional Behavioural Measurement for Black-Box Systems: A Cybernetic Framework for Observable Evaluation"**, modelling system behaviour as empirical distributions and comparing them via calibrated information-theoretic and geometric measures.

---

## Components

### Core Modules

| Module               | Description |
|----------------------|-------------|
| `prepare_samples.py` | Performs the initial sampling of all calibration and test prompts for behavioural evaluation. |
| `sampler.py`         | Generates observable outputs from the `gpt-4o-mini` model (treated as a black-box system). |
| `embedder.py`        | Projects outputs into a measurable observable space (ψ) using a semantic encoder. |
| `divergence.py`      | Computes distributional measures of behavioural difference (Jensen–Shannon, Wasserstein-1, Total Variation). |
| `calibrate.py`       | Estimates intrinsic variability thresholds (θ<sub>D</sub>) from control prompt pairs via bootstrap and percentile analysis. |
| `drift_test.py`      | Implements the **Temporal Drift Validation**, comparing calibrated distributions between temporal checkpoints T₁ and T₂, normalising divergences, and exporting JSON / LaTeX summaries. |
| `run_case.py`        | Executes complete experimental cases, chaining sampling, embedding, divergence computation, calibration, and reporting. |

### Additional Modules

| Module                      | Description |
|-----------------------------|-------------|
| `generate_baseline.py`      | Builds empirical reference thresholds from nominal (stable) behavioural conditions. |
| `analyze_output.py`         | Extracts representative and divergent output pairs for interpretability and qualitative inspection of behavioural regimes. |
| `visualize.py`              | Produces t-SNE and UMAP visualisations of observable distributions, supporting analysis of behavioural proximity and activation regions. |
| `inference_tools.py`        | Implements the Functional Inference Mechanism (FIM) and Lipschitz-Bounded Inference Principle (LBIP) for cross-measure and cross-observable reasoning. |
| `sensitivity_eval.py`       | Performs bootstrap-based sensitivity analysis of distributional measures and classifies the stability of calibrated indicators. |
| `report.py`                 | Summarises numerical and graphical results for each case, generating reproducible tables and figures aligned with the article’s experimental sections. |

---

## Step-by-Step Execution

### 1. Build Baseline (once)

```bash
python generate_baseline.py
```

Establishes the empirical reference thresholds (Î¸<sub>D</sub>) for each behavioural measure under nominal (stable) conditions.  
This step calibrates the systemâ€™s intrinsic variability and must be executed once before any comparative experiment.

---

### 2. Generate Observable Outputs for Calibration or Test Prompts

```bash
python prepare_samples.py
```

Generates responses from the black-box model (`gpt-4o-mini`) for all defined prompt pairs.  
Outputs are stored as raw text for subsequent embedding and divergence computation.

---

### 3. Run a Full Experimental Case

```bash
python run_case.py photosynthesis_gravity
```

You can replace `"photosynthesis_gravity"` with any valid prompt name defined in the `"prompts/"` folder (omit the `.json` extension).  
This command executes the complete behavioural evaluation workflow, including embedding, divergence computation, and calibrated reporting.

---

### 4. Local Output Analysis (optional)

```bash
python analyze_output.py photosynthesis_gravity
```

Performs qualitative inspection of observable differences and representative samples between paired conditions.  
This applies to any configured case inside `"prompts/"` by referencing its filename without the `.json` extension.

---

## Special Experimental Analyses

### Temporal Drift Validation 

```bash
python drift_test.py
```

Compares calibrated behavioural distributions between two temporal checkpoints (Tâ‚ and Tâ‚‚) to assess system homeostasis and behavioural stability.

Outputs include:

- Normalised divergence ratios (R<sub>JS</sub>, R<sub>Wâ‚</sub>)  
- JSON summary with mean activation ratios  
- LaTeX table for direct inclusion in reports

---

### Cross-Observable Inference (LBIP Validation)

No separate execution is required.  
This analysis derives from previously computed calibration and drift data, examining whether behavioural activations in one observable channel proportionally bound those in another, according to the **Lipschitz-Bounded Inference Principle (LBIP)** defined in Section IV of the paper.  
Interpretation is performed directly over the normalised metrics (R<sub>D</sub>) exported from prior runs.

---

### Bootstrap Sensitivity Analysis

```bash
python sensitivity_eval.py
```

Assesses the robustness of the distributional measures under resampling and quantifies the stability of calibration thresholds.

Outputs:

- Confidence intervals for divergence metrics  
- Variability classification  
- Stability visualisations and empirical convergence plots

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

- **Observable outputs (`A_outputs.json`, `B_outputs.json`)** â€“ raw model responses under each input condition.  
- **Embeddings (`A_emb.npy`, `B_emb.npy`)** â€“ vector representations used for behavioural comparison.  
- **Reports (`report.json`)** â€“ calibrated divergence values and normalised ratios (R<sub>D</sub>).  
- **Examples (`examples.json`)** â€“ representative and divergent samples for qualitative inspection.  
- **Visualisations** â€“ kernel density estimates (`kde_distances.png`) and low-dimensional projections (`t-SNE`, `UMAP`) illustrating observable distributions.  
- **Bootstrap results** â€“ variability and stability diagnostics for each measure, including summary statistics and confidence plots.

---


## Contact

For questions, collaborations or academic inquiries:

**Carlos Mario Braga**  
[carlosmario.braga1@alu.uclm.es](mailto:carlosmario.braga1@alu.uclm.es)

