# Introduction
VAND4.0@CVPR2026 Retail Track: Kaputt Defect Detection Challenge: off-the-shelf VLM inference code.

# Dataset
## convert parquet to json format
```bash
python tool/build_dataset_json.py --data_root path/to/k2_datasets --split test --output path/to/dataset_of_json_format.json
```

# Inference
> To reduce token consumption, the experiment is conducted in two stages. For example, material prediction results are obtained first, then used as input for three-stage inference.

## VLM Inference
### Material Prediction
Set in `pipeline.yaml`:

```yaml
data_root: "/Users/hyer/datasets/vand2026/k1/k1_val_sample500"
skip_stage1_and_2: false
material_prediction_only: true  # Set to true to only run Stage 1 and Stage 2
```

```bash
bash run_stage1_and_stage2.sh
```

Output directory structure:
./data/kaputt/${version}/${version}/
    |-- material_predictions.json    // material prediction result


### Execute Inference Based on Material Prediction Results
Set in `pipeline.yaml`:

```yaml
material_prediction_only: false  # Set to true to only run Stage 1 and Stage 2
predicted_material_file: "path/to/material_predictions.json"
skip_stage1_and_2: true  # Set to true to skip material classification and matching verification
```

```bash
bash run_stage3.sh
```

Output directory structure:
./data/kaputt/${version}/${version}/
    |-- ${version}.jsonl    // Detailed inference results
    |-- ${version}.csv      // Prediction results for each capture id with score


### VLM Single-Stage Prediction Result Assistance
We used two models, qwen3.6-27b and GLM 4.6-V, for single-stage prediction
```bash
bash run_single_stage_vlm.sh
```

## Ensemble
Multi-model fusion script supporting two methods:

| Method | Config Value | Description |
|--------|-------------|-------------|
| context-aware | `method: context-aware` | Material defect rate and other prior features |
| dr_adjusted_mean | `method: dr_adjusted_mean` | Defect-rate adjusted mean, scales predictions by material defect rate ratio then averages, no training required |

### Configuration
Edit `configs/ensemble.yaml`:

```yaml
MODEL_REGISTRY:
  "gemini":
    val: ./data/sample/gemini/v17_gemini3-flash-preview_k1_val500.json
    test: ./data/sample/gemini/v17_gemini3-flash-preview_k1_test500.json
  "qw36plus":
    val: ./data/sample/qw36plus/v13_8_val.json
    test: ./data/sample/qw36plus/v13_8_test.json

method: context-aware   # or dr_adjusted_mean
```

### Reproduction
Place the prediction result JSON paths on the val set `data/sample/k1_val_sample500.json` under `val`, and place the prediction JSON paths on the kaputt2 dataset under `test`.
We have already provided the prediction results of corresponding models on the val set in the `data/sample/` directory and configured them in `./configs/ensemble.yaml`. Users need to run inference on kaputt2 themselves. Please place the prediction result JSON paths under `test`.

```yaml
  "model_1":
    val: model_1_val.json
    test: model_1_test.json
  "model_2":
    val: model_2_val.json
    test: model_2_test.json
```

```bash
bash run_ensemble.sh
```

### Usage

```bash
# Use default config configs/ensemble.yaml
python ensemble.py

# Specify config file
python ensemble.py -c configs/ensemble.yaml

# Save predictions to CSV
python ensemble.py -c configs/ensemble.yaml -s output.csv

# No GT on test set, output predictions only
python ensemble.py --no_gt -s output.csv

# Skip LOO contribution analysis
python ensemble.py --no_loo

# Disable material binary features
python ensemble.py --no_hierarchical

# Save / load LR coefficients
python ensemble.py --save_coefs coefs.json
python ensemble.py --load_coefs coefs.json -s output.csv
```
