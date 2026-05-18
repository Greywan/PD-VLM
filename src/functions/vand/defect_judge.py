"""Kaputt retail defect detection via VLM.

Reads query + reference parquet data, sends images to a VLM model,
and outputs defect prediction results per capture.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from datetime import datetime
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tool.call_lm_model import call_api_lm_model_messages, call_api_vlm_model_messages_dashscope
from tool.img_utils import read_img_returnimgobj
from tool.json_process import write_json
from tool.utils import convert_str2dict, extract_defect_and_explain
from tool.prompt_builder import build_prompt

DEFECT_TYPE_MAP = {
    "actuation": "驱动异常",
    "deconstruction": "结构破损",
    "deformation": "变形",
    "missing_unit": "缺失",
    "penetration": "穿透",
    "spillage": "溢出",
    "superficial": "表面缺陷",
}


def translate_defect_types(defect_types_str: str) -> str:
    """Translate comma-separated English defect types to Chinese."""
    if not defect_types_str:
        return ""
    types = [t.strip() for t in defect_types_str.split(",") if t.strip()]
    translated = [DEFECT_TYPE_MAP.get(t, t) for t in types]
    return ",".join(translated)


DEFECT_PROMPT = """你是一个工业质检专家。请对比查询图片（待检物品）和参考图片（正常物品），判断查询图片中的物品是否存在缺陷。

请仔细观察：
1. 查询图片中物品的外观、颜色、形状是否与参考图片一致
2. 是否存在破损、缺失、变形、污染、划痕等缺陷
3. 物品表面是否有异常

请以JSON格式输出结果：
{
    "defect": true或false,
    "explain": "判断理由的简要说明"
}

只输出JSON，不要输出其他内容。"""


def load_kaputt_data(root: str | Path, split: str = "sample"):
    """Load query and reference parquet data from Kaputt track directory.

    Args:
        root: Root directory containing query-{split}.parquet and
            reference-{split}.parquet.
        split: Dataset split name (e.g. "train", "sample").

    Returns:
        Tuple of (query_df, reference_df).
    """
    root = Path(root)
    query_path = root / f"query-{split}.parquet"
    ref_path = root / f"reference-{split}.parquet"

    if not query_path.exists():
        raise FileNotFoundError(f"Query parquet not found: {query_path}")

    query_df = pd.read_parquet(query_path)
    ref_df = pd.read_parquet(ref_path) if ref_path.exists() else pd.DataFrame()
    return query_df, ref_df


def judge_defect_single(
    query_img_path: str,
    ref_img_paths: list[str],
    model_name: str,
    client,
    prompt: str = DEFECT_PROMPT,
) -> dict:
    """Judge whether a query image has defects by comparing with references.

    Args:
        query_img_path: Path to the query image.
        ref_img_paths: List of reference image paths.
        model_name: VLM model name.
        client: OpenAI-compatible client.
        prompt: System/user prompt template.

    Returns:
        Dict with keys: defect (bool), explain (str), raw_response (str).
    """
    # Build multi-image content
    user_content = []

    # Add query image
    _, img_format, base64_img = read_img_returnimgobj(query_img_path)
    if base64_img is None:
        return {"defect": False, "explain": "查询图片读取失败", "raw_response": ""}

    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/{img_format};base64,{base64_img}"},
    })

    # Add reference images
    for ref_path in ref_img_paths:
        _, ref_format, ref_base64 = read_img_returnimgobj(ref_path)
        if ref_base64 is None:
            continue
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{ref_format};base64,{ref_base64}"},
        })

    user_content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": user_content},
    ]

    response_text, cost_time, (prompt_tokens, completion_tokens) = call_api_lm_model_messages(
        model_name=model_name,
        client=client,
        messages=messages,
        response_format="json_object",
    )

    if response_text == "模型调用失败":
        return {"defect": False, "explain": "模型调用失败", "raw_response": ""}

    # Parse response
    try:
        result = json.loads(response_text)
        defect = bool(result.get("defect", False))
        explain = result.get("explain", "")
    except (json.JSONDecodeError, TypeError):
        # Try fallback parsing
        try:
            from tool.utils import convert_str2dict
            result = convert_str2dict(response_text)
            if result and "defect" in result:
                defect = bool(result["defect"])
                explain = result.get("explain", "")
            else:
                raise ValueError("convert_str2dict failed")
        except Exception:
            result = extract_defect_and_explain(response_text)
            if result and "defect" in result:
                defect = bool(result["defect"])
                explain = result.get("explain", "")
            else:
                defect = False
                explain = "无法解析模型输出"

    return {"defect": defect, "explain": explain, "raw_response": response_text}


def vand_defect_predict(information, model_name, config=None, client=None):
    """Single-item defect prediction, compatible with batch processing framework.

    Args:
        information: Dict from dataset.json, containing query_image,
            reference_image list, capture_id, etc.
        model_name: Model name (overridden by config if present).
        config: Config dict with 'data_root' and 'functions/defect' settings.
        output_dir: Output directory (unused, kept for interface compatibility).
        client: OpenAI-compatible client.

    Returns:
        Tuple of (result_map, cost_time, cost_tokens).
    """
    data_root = config.get("data_root", "")
    # model_name = config.get("functions", {}).get("defect", {}).get("model", model_name)
    prompt = build_prompt(config['functions']['description']['user_prompt'])

    model_name = config['functions']['description']['model']
    # Resolve image paths relative to data_root
    query_img_path = str(Path(data_root) / information["query_crop"]) if data_root else information["query_crop"]
    ref_img_paths = [
        str(Path(data_root) / p) if data_root else p
        for p in information.get("reference_crop", [])
    ]

    print("-------------------------------- capture_id --------------------------------")
    print(information.get("capture_id", ""))
    print("-------------------------------- query_crop --------------------------------")
    print(query_img_path)
    print("-------------------------------- reference_crop --------------------------------")

    print("-------------------------------- gt --------------------------------")
    print(information.get("defect", ""))
    print("-------------------------------- major_defect --------------------------------")
    print(information.get("major_defect", ""))

    result_map = {
        "pred": 0,
        "explain": "",
        "capture_id": information.get("capture_id", ""),
        "query_img_name": Path(information["query_image"]).name,
        "reference_img_name": information.get("reference_crop", []),
        "gt_defect": bool(information.get("defect", False)),
        "gt_major_defect": information.get("major_defect", ""),
        "gt_defect_types": information.get("defect_types", ""),
        "item_identifier": information.get("item_identifier", ""),
    }
    
    # Read query image
    image_obj, img_format, base64_img = read_img_returnimgobj(query_img_path)
    if base64_img is None:
        result_map["explain"] = "查询图片读取失败"
        return result_map, 0, (0, 0)

    # Build multi-image content
    user_content = [{
        "type": "image_url",
        "image_url": {"url": f"data:image/{img_format};base64,{base64_img}"},
    }]

    for ref_path in ref_img_paths:
        _, ref_format, ref_base64 = read_img_returnimgobj(ref_path)
        if ref_base64 is None:
            continue
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{ref_format};base64,{ref_base64}"},
        })

    user_content.append({"type": "text", "text": prompt})
    print(user_content)
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": user_content},
    ]

    temperature = config.get("temperature", None) if config else None
    response_text, cost_time, cost_tokens = call_api_lm_model_messages(
        model_name=model_name,
        client=client,
        messages=messages,
        response_format="json_object",
        temperature=temperature,
    )

    print("-------------------------------- 模型输出 --------------------------------")
    print(response_text)

    if response_text == "模型调用失败":
        result_map["explain"] = "模型调用失败"
        return result_map, cost_time, cost_tokens

    # Parse response
    try:
        result = json.loads(response_text)
        # result_map["pred"] = bool(result.get("defect", False))
        result_map["pred"] = float(result.get("defect", 0))
        result_map["explain"] = result.get("explain", "")
    except (json.JSONDecodeError, TypeError):
        result = convert_str2dict(response_text)
        if result and "defect" in result:
            result_map["pred"] = float(result.get("defect", 0))
            result_map["explain"] = result.get("explain", "")
        else:
            result = extract_defect_and_explain(response_text)
            if result and "defect" in result:
                result_map["pred"] = float(result.get("defect", 0))
                result_map["explain"] = result.get("explain", "")
            else:
                result_map["explain"] = "无法解析模型输出"

    return result_map, cost_time, cost_tokens


def vand_defect_predict_ref(information, model_name, config=None, client=None):
    """Single-item defect prediction, compatible with batch processing framework.

    Args:
        information: Dict from dataset.json, containing query_image,
            reference_image list, capture_id, etc.
        model_name: Model name (overridden by config if present).
        config: Config dict with 'data_root' and 'functions/defect' settings.
        output_dir: Output directory (unused, kept for interface compatibility).
        client: OpenAI-compatible client.

    Returns:
        Tuple of (result_map, cost_time, cost_tokens).
    """
    data_root = config.get("data_root", "")
    # model_name = config.get("functions", {}).get("defect", {}).get("model", model_name)
    prompt = build_prompt(config['functions']['description']['user_prompt'])

    model_name = config['functions']['description']['model']
    # Resolve image paths relative to data_root
    query_img_path = str(Path(data_root) / information["query_crop"]) if data_root else information["query_crop"]
    ref_img_paths = [
        str(Path(data_root) / p) if data_root else p
        for p in information.get("reference_crop", [])
    ]

    print("-------------------------------- capture_id --------------------------------")
    print(information.get("capture_id", ""))
    print("-------------------------------- query_crop --------------------------------")
    print(query_img_path)
    print("-------------------------------- reference_crop --------------------------------")
    for p in ref_img_paths:
        print(p)
    print("-------------------------------- gt --------------------------------")
    print(information.get("defect", ""))
    print("-------------------------------- major_defect --------------------------------")
    print(information.get("major_defect", ""))

    result_map = {
        "pred": "",
        "explain": "",
        "capture_id": information.get("capture_id", ""),
        "query_img_name": Path(information["query_image"]).name,
        "img_path": information["query_image"],
        "reference_images": information.get("reference_crop", []),
        "gt_is_defect": bool(information.get("defect", False)),
        "gt_major_defect": information.get("major_defect", ""),
        "gt_defect_types": translate_defect_types(information.get("defect_types", "")),
        "item_identifier": information.get("item_identifier", ""),
        "raw_response": "",
        "timestamp": datetime.now().isoformat(),
        "event": ""
    }
    

    # Build multi-image content
    user_content = [{
        "type": "text",
        "text": config['functions']['description']['refer_prompt_1'],
    }]

    # Read query image
    image_obj, img_format, base64_img, size_info = read_img_returnimgobj(query_img_path)
    if base64_img is None:
        result_map["event"] = "img read failed"
        return result_map, 0, (0, 0)
    if size_info["is_too_small"]:
        result_map["pred"] = 1
        result_map["explain"] = f"img size info:{size_info}，not "
        return result_map, 0, (0, 0)
    
    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/{img_format};base64,{base64_img}"}})

    user_content.append({"type": "text", "text": config['functions']['description']['refer_prompt_2']})
    
    for ref_path in ref_img_paths:
        _, ref_format, ref_base64, size_info = read_img_returnimgobj(ref_path)
        if ref_base64 is None or size_info["is_too_small"]:
            continue
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{ref_format};base64,{ref_base64}"},
        })

    user_content.append({"type": "text", "text": prompt})
    

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": user_content},
    ]

    temperature = config.get("temperature", None) if config else None
    print(f'model_name: {model_name}, temperature: {temperature}')
    response_text, cost_time, cost_tokens = call_api_lm_model_messages(
        model_name=model_name,
        client=client,
        messages=messages,
        response_format=config["functions"]["description"]["output_format"],
    )

    print("-------------------------------- 模型输出 --------------------------------")
    print(response_text)
    result_map["raw_response"] = response_text

    if response_text == "模型调用失败" or len(response_text) == 0:
        result_map["event"] = "模型调用失败"
        return result_map, cost_time, cost_tokens

    # Parse response
    try:
        result = json.loads(response_text)
        # result_map["pred"] = bool(result.get("defect", False))
        result_map["pred"] = float(result.get("defect", 0))
        result_map["explain"] = result.get("explain", "")
    except (json.JSONDecodeError, TypeError):
        result = convert_str2dict(response_text)
        if result and "defect" in result:
            result_map["pred"] = float(result.get("defect", 0))
            result_map["explain"] = result.get("explain", "")
        else:
            result = extract_defect_and_explain(response_text)
            if result and "defect" in result:
                result_map["pred"] = float(result.get("defect", 0))
                result_map["explain"] = result.get("explain", "")
            else:
                result_map["event"] = "无法解析模型输出"

    return result_map, cost_time, cost_tokens


def vand_defect_predict_interwoven(information, model_name, config=None, client=None):
    """Single-item defect prediction, compatible with batch processing framework.

    Args:
        information: Dict from dataset.json, containing query_image,
            reference_image list, capture_id, etc.
        model_name: Model name (overridden by config if present).
        config: Config dict with 'data_root' and 'functions/defect' settings.
        output_dir: Output directory (unused, kept for interface compatibility).
        client: OpenAI-compatible client.

    Returns:
        Tuple of (result_map, cost_time, cost_tokens).
    """
    data_root = config.get("data_root", "")
    # model_name = config.get("functions", {}).get("defect", {}).get("model", model_name)

    model_name = config['functions']['description']['model']
    # Resolve image paths relative to data_root
    query_img_path = str(Path(data_root) / information["query_crop"]) if data_root else information["query_crop"]
    ref_img_paths = [
        str(Path(data_root) / p) if data_root else p
        for p in information.get("reference_crop", [])
    ]

    print("-------------------------------- capture_id --------------------------------")
    print(information.get("capture_id", ""))
    print("-------------------------------- query_crop --------------------------------")
    print(query_img_path)
    print("-------------------------------- reference_crop --------------------------------")
    for p in ref_img_paths:
        print(p)
    print("-------------------------------- gt --------------------------------")
    print(information.get("defect", ""))
    print("-------------------------------- major_defect --------------------------------")
    print(information.get("major_defect", ""))

    result_map = {
        "pred": "",
        "explain": "",
        "capture_id": information.get("capture_id", ""),
        "query_img_name": Path(information["query_image"]).name,
        "img_path": information["query_image"],
        "reference_images": information.get("reference_crop", []),
        "gt_is_defect": bool(information.get("defect", False)),
        "gt_major_defect": information.get("major_defect", ""),
        "gt_defect_types": translate_defect_types(information.get("defect_types", "")),
        "item_identifier": information.get("item_identifier", ""),
        "raw_response": "",
        "timestamp": datetime.now().isoformat(),
        "event": ""
    }
    

    # Build multi-image content
    user_content = [{
        "type": "text",
        "text": config['functions']['description']['query_prompt_1'],
    }]

    # Read query image
    image_obj, img_format, base64_img = read_img_returnimgobj(query_img_path)
    if base64_img is None:
        result_map["event"] = "查询图片读取失败"
        return result_map, 0, (0, 0)
    
    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/{img_format};base64,{base64_img}"}})

    user_content.append({"type": "text", "text": config['functions']['description']['refer_prompt_1']})
    
    for ref_path in ref_img_paths:
        _, ref_format, ref_base64 = read_img_returnimgobj(ref_path)
        if ref_base64 is None:
            continue
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{ref_format};base64,{ref_base64}"},
        })

    prompt = build_prompt(config['functions']['description']['user_prompt'])
    user_content.append({"type": "text", "text": prompt})
    

    messages = [
        {"role": "system", "content": [{"type": "text", "text": config['functions']['description']['system_prompt']}]},
        {"role": "user", "content": user_content},
    ]

    temperature = config.get("temperature", 0.01) if config else None
    print(f'model_name: {model_name}, temperature: {temperature}')
    response_text, cost_time, cost_tokens = call_api_lm_model_messages(
        model_name=model_name,
        client=client,
        messages=messages,
        response_format=config["functions"]["description"]["output_format"],
    )

    print("-------------------------------- 模型输出 --------------------------------")
    print(response_text)
    result_map["raw_response"] = response_text

    if response_text == "模型调用失败" or len(response_text) == 0:
        result_map["event"] = "模型调用失败"
        return result_map, cost_time, cost_tokens

    # Parse response
    try:
        result = json.loads(response_text)
        # result_map["pred"] = bool(result.get("defect", False))
        result_map["pred"] = float(result.get("defect", 0))
        result_map["explain"] = result.get("explain", "")
    except (json.JSONDecodeError, TypeError):
        result = convert_str2dict(response_text)
        if result and "defect" in result:
            result_map["pred"] = float(result.get("defect", 0))
            result_map["explain"] = result.get("explain", "")
        else:
            result = extract_defect_and_explain(response_text)
            if result and "defect" in result:
                result_map["pred"] = float(result.get("defect", 0))
                result_map["explain"] = result.get("explain", "")
            else:
                result_map["event"] = "无法解析模型输出"

    return result_map, cost_time, cost_tokens

def run_defect_detection(
    data_root: str | Path,
    split: str = "sample",
    model_name: str = "qwen2.5-vl-72b-instruct",
    client=None,
    output_path: str | Path | None = None,
    max_samples: int | None = None,
):
    """Run defect detection on Kaputt data and output results.

    Args:
        data_root: Root directory with parquet files and image data.
        split: Dataset split name.
        model_name: VLM model to use.
        client: OpenAI-compatible client instance.
        output_path: Path to save results JSON. If None, prints only.
        max_samples: Max number of samples to process. None for all.
    """
    data_root = Path(data_root)
    query_df, ref_df = load_kaputt_data(data_root, split)

    if max_samples:
        query_df = query_df.head(max_samples)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, row in tqdm(query_df.iterrows(), total=len(query_df), desc="Defect detection"):
        capture_id = row.capture_id
        item_identifier = row.item_identifier

        # Resolve image paths relative to data_root
        query_img_path = str(data_root / row.query_image)
        query_img_name = Path(row.query_image).name

        # Collect reference images for this item
        ref_rows = ref_df[ref_df.item_identifier == item_identifier] if len(ref_df) > 0 else pd.DataFrame()
        ref_img_paths = [str(data_root / r.reference_image) for _, r in ref_rows.iterrows()]
        ref_img_names = [Path(r.reference_image).name for _, r in ref_rows.iterrows()]

        print(f"\n--- capture_id: {capture_id} ---")
        print(f"  query: {query_img_name}")
        print(f"  refs: {ref_img_names}")

        result = judge_defect_single(query_img_path, ref_img_paths, model_name, client)

        results.append({
            "idx": idx,
            "capture_id": capture_id,
            "pred": result["defect"],
            "query_img_name": query_img_name,
            "reference_img_name": ", ".join(ref_img_names),
            "explain": result["explain"],
            "gt_defect": bool(row.get("defect", False)),
            "gt_major_defect": row.get("major_defect", ""),
            "gt_defect_types": translate_defect_types(row.get("defect_types", "")),
            "item_identifier": item_identifier,
        })

        print(f"pred: {result['defect']}, gt: {bool(row.get('defect', False))}, explain: {result['explain']}")
        # Save
        if output_path:
            write_json(str(output_path), results)
            
    # Summary
    if results:
        correct = sum(1 for r in results if r["pred"] == r["gt_defect"])
        total = len(results)
        print(f"\n===== 结果统计 =====")
        print(f"总数: {total}, 正确: {correct}, 准确率: {correct/total:.2%}")

    

    return results


def vand_defect_predict_dashscope(information, model_name, config=None, client=None):
    """Single-item defect prediction, compatible with batch processing framework.

    Args:
        information: Dict from dataset.json, containing query_image,
            reference_image list, capture_id, etc.
        model_name: Model name (overridden by config if present).
        config: Config dict with 'data_root' and 'functions/defect' settings.
        output_dir: Output directory (unused, kept for interface compatibility).
        client: OpenAI-compatible client.

    Returns:
        Tuple of (result_map, cost_time, cost_tokens).
    """
    data_root = config.get("data_root", "")
    # model_name = config.get("functions", {}).get("defect", {}).get("model", model_name)
    prompt = build_prompt(config['functions']['description']['user_prompt'])

    model_name = config['functions']['description']['model']
    # Resolve image paths relative to data_root
    query_img_path = str(Path(data_root) / information["query_crop"]) if data_root else information["query_crop"]
    ref_img_paths = [
        str(Path(data_root) / p) if data_root else p
        for p in information.get("reference_crop", [])
    ]

    print("-------------------------------- capture_id --------------------------------")
    print(information.get("capture_id", ""))
    print("-------------------------------- query_crop --------------------------------")
    print(query_img_path)
    print("-------------------------------- reference_crop --------------------------------")
    for p in ref_img_paths:
        print(p)
    print("-------------------------------- gt --------------------------------")
    print(information.get("defect", ""))
    print("-------------------------------- major_defect --------------------------------")
    print(information.get("major_defect", ""))

    result_map = {
        "pred": "",
        "explain": "",
        "capture_id": information.get("capture_id", ""),
        "query_img_name": Path(information["query_image"]).name,
        "img_path": information["query_image"],
        "reference_images": information.get("reference_crop", []),
        "gt_is_defect": bool(information.get("defect", False)),
        "gt_major_defect": information.get("major_defect", ""),
        "gt_defect_types": translate_defect_types(information.get("defect_types", "")),
        "item_identifier": information.get("item_identifier", ""),
        "raw_response": "",
        "timestamp": datetime.now().isoformat(),
        "event": ""
    }
    

    # Build multi-image content
    user_content = [{
        "text": config['functions']['description']['refer_prompt_1'],
    }]

    # Read query image
    image_obj, img_format, base64_img = read_img_returnimgobj(query_img_path)
    if base64_img is None:
        result_map["explain"] = "查询图片读取失败"
        return result_map, 0, (0, 0)
    
    user_content.append({"image": f"data:image/{img_format};base64,{base64_img}"})

    user_content.append({"text": config['functions']['description']['refer_prompt_2']})
    
    for ref_path in ref_img_paths:
        _, ref_format, ref_base64 = read_img_returnimgobj(ref_path)
        if ref_base64 is None:
            continue
        user_content.append({
            "image": f"data:image/{img_format};base64,{base64_img}",
        })

    user_content.append({"text": prompt})
    

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]
    api_key = config["DASHSCOPE_API_KEY"]
    response_text, cost_time, cost_tokens = call_api_vlm_model_messages_dashscope(
        model_name=model_name,
        api_key=api_key,
        messages=messages,
        response_format=config["functions"]["description"]["output_format"],
    )

    print("-------------------------------- 模型输出 --------------------------------")
    print(response_text)
    result_map["raw_response"] = response_text

    if response_text == "模型调用失败" or len(response_text) == 0:
        result_map["event"] = "模型调用失败"
        return result_map, cost_time, cost_tokens

    # Parse response
    try:
        result = json.loads(response_text)
        # result_map["pred"] = bool(result.get("defect", False))
        result_map["pred"] = float(result.get("defect", 0))
        result_map["explain"] = result.get("explain", "")
    except (json.JSONDecodeError, TypeError):
        result = convert_str2dict(response_text)
        if result and "defect" in result:
            result_map["pred"] = float(result.get("defect", 0))
            result_map["explain"] = result.get("explain", "")
        else:
            result = extract_defect_and_explain(response_text)
            if result and "defect" in result:
                result_map["pred"] = float(result.get("defect", 0))
                result_map["explain"] = result.get("explain", "")
            else:
                result_map["event"] = "无法解析模型输出"

    return result_map, cost_time, cost_tokens


def vand_defect_predict_pipeline(information, model_name, config=None, client=None):
    """三阶段缺陷检测分析流程。

    Stage 1: 提示词选择器 - 对query图片进行材质分类
    Stage 2: 提示词匹配度判别器 - 验证材质分类是否匹配
    Stage 3: 提示词分析执行器 - 使用对应材质的提示词进行缺陷分析

    Args:
        information: Dict from dataset.json, containing query_image,
            reference_image list, capture_id, etc.
        model_name: Model name.
        config: Config dict with pipeline settings and prompt paths.
        client: OpenAI-compatible client.

    Returns:
        Tuple of (result_map, cost_time, cost_tokens).
    """
    import yaml
    from tool.prompt_builder import build_prompt

    data_root = config.get("data_root", "")
    model_name = config['functions']['description']['model']
    temperature = config.get("temperature", 0.01)

    # 读取跳过 Stage 1/2 的配置
    skip_stage1_and_2 = config.get("skip_stage1_and_2", False)
    material_source_field = config.get("material_source_field", "item_material")
    target_materials = config.get("target_materials", [])

    # 材质预测模式（只做 Stage 1 和 Stage 2，输出材质预测结果）
    material_prediction_only = config.get("material_prediction_only", False)
    material_prediction_output = config.get("material_prediction_output", None)

    # 材质映射表（中文 -> 英文代码）
    MATERIAL_MAPPING = {
        "纸张": "paper",
        "硬纸板/纸箱": "cardboard",
        "纸质书籍": "book_paper",
        "塑封书籍": "book_plastic_tight_wrap",
        "书籍": "book",
        "其他书籍": "book_other",
        "塑料松散袋": "plastic_loose_bag",
        "塑料紧包装/塑封": "plastic_tight_wrap",
        "塑料气泡膜": "plastic_bubble_wrap",
        "硬塑料": "plastic_hard",
        "其他": "other"
    }

    # 反向映射表（英文代码 -> 中文）
    REVERSE_MATERIAL_MAPPING = {v: k for k, v in MATERIAL_MAPPING.items()}

    # 加载预测材质文件（如果有）
    predicted_material_file = config.get("predicted_material_file", None)
    predicted_materials = {}
    if predicted_material_file and Path(predicted_material_file).exists():
        with open(predicted_material_file, 'r', encoding='utf-8') as f:
            predicted_materials = json.load(f)
        # 如果文件嵌套在 "query" 键下，提取出来
        if "query" in predicted_materials:
            predicted_materials = predicted_materials["query"]
        print(f"已加载预测材质文件: {predicted_material_file}, 共 {len(predicted_materials)} 条记录")

    # 加载提示词配置
    base_prompt_path = config.get("base_prompt_path", "configs/base_prompt.yaml")
    material_prompts_path = config.get("material_prompts_path", "configs/material_prompts.yaml")

    with open(base_prompt_path, 'r', encoding='utf-8') as f:
        base_prompts = yaml.safe_load(f)
    with open(material_prompts_path, 'r', encoding='utf-8') as f:
        material_prompts = yaml.safe_load(f)

    # 如果设置了 target_materials，检查当前样本是否需要分析
    if target_materials and skip_stage1_and_2:
        # 从输入数据读取材质
        sample_material = information.get(material_source_field, "")
        if sample_material not in target_materials:
            print(f"跳过样本: material '{sample_material}' 不在目标材质列表 {target_materials} 中")
            # 返回空结果，表示跳过此样本
            return None, 0, (0, 0)

    # 如果设置了 target_materials 但没有 skip_stage1_and_2，需要在 Stage 1 后过滤
    # 这种情况会在 Stage 1 完成后检查

    # Resolve image paths
    query_img_path = str(Path(data_root) / information["query_crop"]) if data_root else information["query_crop"]
    ref_img_paths = [
        str(Path(data_root) / p) if data_root else p
        for p in information.get("reference_crop", [])
    ]

    print("=" * 60)
    print(f"【三阶段缺陷检测】capture_id: {information.get('capture_id', '')}")
    print(f"query: {query_img_path}")
    print("=" * 60)

    # 初始化结果map
    result_map = {
        "pred": 0.0,
        "explain": "",
        "capture_id": information.get("capture_id", ""),
        "query_img_name": Path(information["query_image"]).name,
        "img_path": information["query_image"],
        "reference_images": information.get("reference_crop", []),
        "gt_is_defect": bool(information.get("defect", False)),
        "gt_major_defect": information.get("major_defect", ""),
        "gt_defect_types": translate_defect_types(information.get("defect_types", "")),
        "item_identifier": information.get("item_identifier", ""),
        "raw_response": "",
        "timestamp": datetime.now().isoformat(),
        "event": "",
        "pipeline_info": {}  # 存储三阶段信息
    }

    # Read query image
    image_obj, img_format, base64_img, size_info = read_img_returnimgobj(query_img_path)
    if base64_img is None:
        result_map["event"] = "img read failed"
        return result_map, 0, (0, 0)

    if size_info["is_too_small"]:
        # 材质预测模式下，图片过小时默认预测为 "other"
        if material_prediction_only:
            result_map["pred"] = -1
            result_map["material_prediction"] = {
                "chinese": "其他",
                "english": "other",
                "stage1_material": "other",
                "stage1_confidence": 1.0,
                "stage2_is_match": True,
                "stage2_match_score": 1.0,
                "note": f"图片尺寸过小: {size_info['new_width']}x{size_info['new_height']}，默认预测为 other"
            }
            result_map["explain"] = f"图片尺寸过小，默认预测材质为 other"
            print(f"[警告] 图片尺寸过小，默认预测材质为 other: {size_info}")
            return result_map, 0, (0, 0)
        else:
            result_map["pred"] = 0
            result_map["explain"] = f"图片信息:{size_info}，不满足要求"
            return result_map, 0, (0, 0)
    # ============================================================
    # Stage 1: 提示词选择器 (Material Code Selector)
    # ============================================================

    if skip_stage1_and_2:
        # 跳过 Stage 1，直接从输入数据读取材质
        print("\n>>> Stage 1: 跳过（从输入数据读取材质）")

        # 优先从预测材质文件中读取
        capture_id = information.get("capture_id", "")
        if capture_id in predicted_materials:
            chinese_material = predicted_materials[capture_id]
            material_code = MATERIAL_MAPPING.get(chinese_material, "other")
            print(f"从预测文件读取材质: '{chinese_material}' -> '{material_code}'")
        else:
            material_code = information.get(material_source_field, "other")

        # 应用材质代码映射（归一化）
        material_code_mapping = base_prompts.get('material_code_mapping', {})
        if material_code in material_code_mapping:
            original_code = material_code
            material_code = material_code_mapping[material_code]
            print(f"材质代码映射: '{original_code}' -> '{material_code}'")

        # 验证material_code是否有效
        valid_codes = base_prompts.get('MATERIAL_CODES', [])
        if material_code not in valid_codes:
            print(f"警告: 无效的material_code '{material_code}'，使用 'other'")
            material_code = "other"

        print(f"Stage 1 结果: material_code={material_code} (从输入数据读取)")

        result_map["pipeline_info"]["stage1"] = {
            "material_code": material_code,
            "confidence": 1.0,
            "reasoning": "跳过Stage 1，从输入数据读取",
            "skipped": True
        }

        # 跳过Stage的成本为0
        cost_time_1 = 0
        cost_tokens_1 = (0, 0)
        response_text_1 = ""
    else:
        print("\n>>> Stage 1: 提示词选择器 - 材质分类")

        stage1_system = base_prompts['stage1_selector']['system_prompt']
        stage1_user = base_prompts['stage1_selector']['user_prompt']

        stage1_messages = [
            {"role": "system", "content": [{"type": "text", "text": stage1_system}]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/{img_format};base64,{base64_img}"}},
                {"type": "text", "text": stage1_user}
            ]},
        ]

        response_text_1, cost_time_1, cost_tokens_1 = call_api_lm_model_messages(
            model_name=model_name,
            client=client,
            messages=stage1_messages,
            response_format="json_object",
            temperature=temperature,
        )

        print(f"Stage 1 输出: {response_text_1}")

        try:
            stage1_result = json.loads(response_text_1)
            material_code = stage1_result.get("material_code", "other")
            stage1_confidence = stage1_result.get("confidence", 0.0)
            stage1_reasoning = stage1_result.get("reasoning", "")
        except (json.JSONDecodeError, TypeError):
            # 解析失败，使用默认值
            material_code = "other"
            stage1_confidence = 0.0
            stage1_reasoning = "解析失败"

        # 应用材质代码映射（归一化）
        material_code_mapping = base_prompts.get('material_code_mapping', {})
        if material_code in material_code_mapping:
            original_code = material_code
            material_code = material_code_mapping[material_code]
            print(f"材质代码映射: '{original_code}' -> '{material_code}'")

        # 验证material_code是否有效
        valid_codes = base_prompts.get('MATERIAL_CODES', [])
        if material_code not in valid_codes:
            print(f"警告: 无效的material_code '{material_code}'，使用 'other'")
            material_code = "other"

        print(f"Stage 1 结果: material_code={material_code}, confidence={stage1_confidence}")

        result_map["pipeline_info"]["stage1"] = {
            "material_code": material_code,
            "confidence": stage1_confidence,
            "reasoning": stage1_reasoning,
            "skipped": False
        }

        # 如果设置了 target_materials 且没有 skip_stage1_and_2，在 Stage 1 后检查
        if target_materials and material_code not in target_materials:
            print(f"跳过样本: 检测到的 material '{material_code}' 不在目标材质列表 {target_materials} 中")
            return None, cost_time_1, cost_tokens_1

    # ============================================================
    # Stage 2: 提示词匹配度判别器 (Prompt Matcher)
    # ============================================================

    if skip_stage1_and_2:
        # 跳过 Stage 2，假设材质匹配
        print(f"\n>>> Stage 2: 跳过（假设材质匹配）")
        is_match = True
        match_score = 1.0
        stage2_reasoning = "跳过Stage 2，假设输入的材质代码准确"
        fallback_code = material_code
        final_material_code = material_code

        print(f"Stage 2 结果: is_match={is_match}, match_score={match_score} (跳过)")

        result_map["pipeline_info"]["stage2"] = {
            "is_match": is_match,
            "match_score": match_score,
            "reasoning": stage2_reasoning,
            "fallback_code": fallback_code,
            "final_material_code": final_material_code,
            "skipped": True
        }

        # 跳过Stage的成本为0
        cost_time_2 = 0
        cost_tokens_2 = (0, 0)
        response_text_2 = ""
    else:
        print(f"\n>>> Stage 2: 提示词匹配度判别器 - 验证 '{material_code}' 是否匹配")

        material_descriptions = base_prompts.get('material_descriptions', {})
        material_description = material_descriptions.get(material_code, "未知材质")

        stage2_system = base_prompts['stage2_matcher']['system_prompt']
        stage2_user_template = base_prompts['stage2_matcher']['user_prompt']
        stage2_user = stage2_user_template.format(
            material_code=material_code,
            material_description=material_description
        )

        stage2_messages = [
            {"role": "system", "content": [{"type": "text", "text": stage2_system}]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/{img_format};base64,{base64_img}"}},
                {"type": "text", "text": stage2_user}
            ]},
        ]

        response_text_2, cost_time_2, cost_tokens_2 = call_api_lm_model_messages(
            model_name=model_name,
            client=client,
            messages=stage2_messages,
            response_format="json_object",
            temperature=temperature,
        )

        print(f"Stage 2 输出: {response_text_2}")

        try:
            stage2_result = json.loads(response_text_2)
            is_match = stage2_result.get("is_match", False)
            match_score = stage2_result.get("match_score", 0.0)
            stage2_reasoning = stage2_result.get("reasoning", "")
            fallback_code = stage2_result.get("fallback_code", "other")
        except (json.JSONDecodeError, TypeError):
            is_match = False
            match_score = 0.0
            stage2_reasoning = "解析失败"
            fallback_code = "other"

        print(f"Stage 2 结果: is_match={is_match}, match_score={match_score}")

        # 如果不匹配，使用fallback_code
        final_material_code = material_code if is_match else fallback_code
        if not is_match:
            print(f"匹配失败，切换到兜底材质: {final_material_code}")

        result_map["pipeline_info"]["stage2"] = {
            "is_match": is_match,
            "match_score": match_score,
            "reasoning": stage2_reasoning,
            "fallback_code": fallback_code,
            "final_material_code": final_material_code,
            "skipped": False
        }

    # ============================================================
    # 材质预测模式：只做 Stage 1 和 Stage 2，保存结果并返回
    # ============================================================
    if material_prediction_only:
        # 将英文材质代码转换为中文
        chinese_material = REVERSE_MATERIAL_MAPPING.get(final_material_code, "其他")

        # 汇总成本
        total_cost_time = cost_time_1 + cost_time_2
        total_tokens = (cost_tokens_1[0] + cost_tokens_2[0],
                       cost_tokens_1[1] + cost_tokens_2[1])

        print(f"\n<<< 材质预测完成（跳过 Stage 3）")
        print(f"    预测材质: {chinese_material} ({final_material_code})")
        print(f"    总耗时: {total_cost_time:.2f}s")
        print(f"    总token: {total_tokens}")
        print("=" * 60)

        # 返回材质预测结果，标记为预测模式
        result_map["pred"] = -1  # 特殊标记，表示这是材质预测，不是缺陷分数
        result_map["material_prediction"] = {
            "chinese": chinese_material,
            "english": final_material_code,
            "stage1_material": result_map["pipeline_info"]["stage1"]["material_code"],
            "stage1_confidence": result_map["pipeline_info"]["stage1"]["confidence"],
            "stage2_is_match": result_map["pipeline_info"]["stage2"]["is_match"],
            "stage2_match_score": result_map["pipeline_info"]["stage2"]["match_score"],
        }

        return result_map, total_cost_time, total_tokens

    # ============================================================
    # Stage 3: 提示词分析执行器 (Analysis Executor)
    # ============================================================
    print(f"\n>>> Stage 3: 提示词分析执行器 - 使用 '{final_material_code}' 进行分析")

    # 获取材质特定的分析提示词
    material_config = material_prompts.get(final_material_code, material_prompts.get("other", {}))
    material_analysis_prompt = material_config.get("analysis_prompt", "")

    # 获取输出格式提示词
    output_format_prompt = base_prompts['stage3_executor']['output_format_prompt']

    # 构建完整提示词
    full_prompt = material_analysis_prompt + "\n" + output_format_prompt

    # 构建多图片消息（query + references）
    user_content = [{"type": "text", "text": "请分析以下商品图片的缺陷情况：\n\n这是【查询图片】（待检物品）："}]
    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/{img_format};base64,{base64_img}"}})

    # 添加参考图片
    if ref_img_paths:
        user_content.append({"type": "text", "text": "\n以下是【参考图片】（正常物品样本）："})
        for ref_path in ref_img_paths:
            _, ref_format, ref_base64, size_info = read_img_returnimgobj(ref_path)
            if ref_base64 is None or size_info["is_too_small"]:
                continue
            if ref_base64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{ref_format};base64,{ref_base64}"},
                })

    user_content.append({"type": "text", "text": "\n" + full_prompt})

    # Stage 3 system prompt 强调严格遵守判定规则
    stage3_system_prompt = """你是一名专业的物流和零售商品质量检验专家。

【核心原则】
1. **与参考图对比是首要依据**：如果查询图与参考图看起来一样，说明是正常状态
2. **检测真实缺陷**：角部翘起、折痕、变形等实质性损坏必须被检测出来
3. **区分正常与缺陷**：
   - 正常：书页厚度导致的整体轻微抬高、软质封面的自然弧度
   - 缺陷：角部实质性的向上翘起、向内折叠、弯曲变形

【判定优先级规则】
1. **首先应用材质特定规则**：上述【领域知识】和【特别说明】中明确排除的情况（如气泡膜褶皱、塑料袋褶皱、标签差异等）**不属于缺陷**，应判定为 defect = 0.0
2. **然后应用强制执行规则**：对于确实属于缺陷的情况（未被特别说明排除的），必须严格遵守以下规则：
   - 发现实质性缺陷（磨损、划痕、污渍、变形、折痕、边缘翘起等），defect得分必须≥0.6
   - 禁止给出0.1-0.5的分数
   - 完全无实质性缺陷时输出 defect = 0.0

【判定流程】
Step 1: 对比参考图 - 如果查询图与参考图状态一致 → 正常状态
Step 2: 根据【缺陷检查清单】逐一检查，特别注意常见缺陷位置
Step 3: 如果没有发现实质性缺陷 → defect = 0.0
Step 4: 如果发现实质性缺陷 → 根据严重程度给出 ≥0.6 的分数

【书籍类重点检查 - 右下角是常见缺陷位置】
- **必须放大检查右下角**：书籍右下角翘起是最常见的缺陷
- **如何判断右下角缺陷**：
  - 正常：边缘整齐，即使有轻微抬高也是整体性的
  - 缺陷：角部有实质性的向上翘起、向内折叠、或明显弯曲变形

【分数标准】
- 0.0 = 无实质性缺陷
- ≥0.6 = 实质性轻微缺陷
- ≥0.8 = 明显缺陷
- =1.0 = 完全损坏"""

    stage3_messages = [
        {"role": "system", "content": [{"type": "text", "text": stage3_system_prompt}]},
        {"role": "user", "content": user_content},
    ]

    response_text_3, cost_time_3, cost_tokens_3 = call_api_lm_model_messages(
        model_name=model_name,
        client=client,
        messages=stage3_messages,
        response_format="json_object",
        temperature=temperature,
    )

    print(f"Stage 3 输出: {response_text_3}")
    result_map["raw_response"] = response_text_3

    # 解析最终结果
    try:
        stage3_result = json.loads(response_text_3)
        result_map["pred"] = float(stage3_result.get("defect", 0))
        result_map["explain"] = stage3_result.get("explain", "")
        item_material = stage3_result.get("item_material", final_material_code)
    except (json.JSONDecodeError, TypeError):
        result = convert_str2dict(response_text_3)
        if result and "defect" in result:
            result_map["pred"] = float(result.get("defect", 0))
            result_map["explain"] = result.get("explain", "")
        else:
            result = extract_defect_and_explain(response_text_3)
            if result and "defect" in result:
                result_map["pred"] = float(result.get("defect", 0))
                result_map["explain"] = result.get("explain", "")
            else:
                result_map["event"] = "无法解析模型输出"
                item_material = final_material_code

    result_map["pipeline_info"]["stage3"] = {
        "material_code_used": final_material_code,
        "item_material_detected": item_material
    }

    # 汇总成本
    total_cost_time = cost_time_1 + cost_time_2 + cost_time_3
    total_tokens = (cost_tokens_1[0] + cost_tokens_2[0] + cost_tokens_3[0],
                   cost_tokens_1[1] + cost_tokens_2[1] + cost_tokens_3[1])

    print(f"\n<<< 三阶段分析完成")
    print(f"    总耗时: {total_cost_time:.2f}s")
    print(f"    总token: {total_tokens}")
    print(f"    预测结果: pred={result_map['pred']}, explain={result_map['explain'][:50]}...")
    print("=" * 60)

    return result_map, total_cost_time, total_tokens


if __name__ == "__main__":
    from tool.options import load_config
    import argparse

    parser = argparse.ArgumentParser(description="Kaputt defect detection via VLM")
    parser.add_argument("--data_root", type=str, default="./data/kaputt/sample-data")
    parser.add_argument("--split", type=str, default="sample")
    parser.add_argument("--model", type=str, default="qwen2.5-vl-72b-instruct")
    parser.add_argument("--output", type=str, default="./outputs/vand_defect_results.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    # Init client
    from openai import OpenAI
    # client = OpenAI(
    #     api_key=os.environ.get("OPENAI_API_KEY", "Bearer sk-xxx"),
    #     base_url=os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible_mode/v1"),
    # )

    client = OpenAI(
        api_key="Bearer sk-xxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    model_name = args.model
    if args.config:
        config = load_config(args.config)
        if config and "model" in config:
            model_name = config["model"]

    run_defect_detection(
        data_root=args.data_root,
        split=args.split,
        model_name=model_name,
        client=client,
        output_path=args.output,
        max_samples=args.max_samples,
    )