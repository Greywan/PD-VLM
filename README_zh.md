# Introduction
VAND4.0@CVPR2026 Retail Track: Kaputt Defect Detection Challenge: off-the-shelf VLM 推理代码。

# Dataset
## convert parquet to json format
```bash
python tool/build_dataset_json.py --data_root path/to/k2_datasets --split test --output path/to/dataset_of_json_format.json
```

# Inference
> 为了减少token消耗，实验过程中分两个阶段进行。例如先得到材质预测结果后，反复作为三阶段推理的输入。

## VLM inference
### 材质预测
`pipeline.yaml`中设置

```yaml
data_root: "/Users/hyer/datasets/vand2026/k1/k1_val_sample500"
skip_stage1_and_2: false
material_prediction_only: true  # 设置为 true 只做 Stage 1 和 Stage 2
```

```bash
bash run_stage1_and_stage2.sh
```

输出目录类似：
./data/kaputt/${version}/${version}/
    |-- material_predictions.json    // material prediction result


### 根据材质预测结果执行推理
`pipeline.yaml`中设置

```yaml
material_prediction_only: false  # 设置为 true 只做 Stage 1 和 Stage 2
predicted_material_file: "path/to/material_predictions.json"
skip_stage1_and_2: true  # 设置为 true 跳过材质分类和匹配验证
```

```bash
bash run_stage3.sh
```

### VLM 单阶段预测结果辅助
我们使用了 qwen3.6-27b 和 GLM 4.6-V 两个模型进行单阶段预测
```bash
bash run_single_stage_vlm.sh
```

输出目录类似：
./data/kaputt/${version}/${version}/
    |-- ${version}.jsonl    // detail inference result
    |-- ${version}.csv      // prediction result of each capture id with score


## ensemble
多模型融合脚本，支持两种方法：

| 方法 | 配置值 | 说明 |
|------|--------|------|
| context-aware | `method: context-aware` | 利用材质缺陷率等先验特征调整 |
| dr_adjusted_mean | `method: dr_adjusted_mean` | 缺陷率调整均值，按材质缺陷率缩放预测后取均值，无需训练 |

### 配置文件
编辑 `configs/ensemble.yaml`：

```yaml
MODEL_REGISTRY:
  "gemini":
    val: ./data/sample/gemini/v17_gemini3-flash-preview_k1_val500.json
    test: ./data/sample/gemini/v17_gemini3-flash-preview_k1_test500.json
  "qw36plus":
    val: ./data/sample/qw36plus/v13_8_val.json
    test: ./data/sample/qw36plus/v13_8_test.json

method: context-aware   # 或 dr_adjusted_mean
```

### 复现
需要在 val 集 `data/sample/k1_val_sample500.json`上预测的结果 json 路径放置在 val 上, 然后将在 kaputt2 数据上预测的 json 路径放置在 test 上
我们已经在 `data/sample/` 目录下提供了对应模型在 val 集上的预测结果，已经在 ./configs/ensembel.yaml 中配置了，需要用户自己在 kaputt2 上运行推理。请将预测结果 json 路径放置在 test 上
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

### 运行

```bash
# 使用默认配置 configs/ensemble.yaml
python ensemble.py

# 指定配置文件
python ensemble.py -c configs/ensemble.yaml

# 保存预测结果到 CSV
python ensemble.py -c configs/ensemble.yaml -s output.csv

# 测试集无 GT，仅输出预测
python ensemble.py --no_gt -s output.csv

# 跳过 LOO 贡献度分析
python ensemble.py --no_loo

# 去掉材质二值特征
python ensemble.py --no_hierarchical

# 保存 / 加载 LR 系数
python ensemble.py --save_coefs coefs.json
python ensemble.py --load_coefs coefs.json -s output.csv
```
