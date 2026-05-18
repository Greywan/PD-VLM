"""
Generic batch processing utility - replaces repetitive *_batch_predict functions
"""
from tqdm import tqdm
from .json_process import write_json, append_jsonl
import json
import logging
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)
from tool.process_table_class import ExcelProcessor


def _call_with_timeout(func, timeout, *args, **kwargs):
    """Function call with timeout, degenerates to normal call when timeout=None"""
    if timeout is None:
        return func(*args, **kwargs)
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)

excel_processor = ExcelProcessor()

def batch_predict(
    data,
    existing_data,
    predictor_func,
    save_path,
    idx_field="idx",
    config=None,
    client=None,
    max_count=None,
    skip_processed=True,
    fields_map=None,
    track_tokens=False,
    token_fields=None,
    extra_fields_func=None,
    max_workers=1,
    item_timeout=20,
    max_retries=2,
):
    """
    Generic batch prediction function

    Args:
        data: Data list to predict
        existing_data: Already processed data list (for resume from checkpoint)
        predictor_func: Prediction function, signature: (item, model_name, client, config) -> (result, cost_time, process_tokens)
        save_path: Save path
        idx_field: Index field name, default "idx"
        max_count: Maximum processing count, None means unlimited
        skip_processed: Whether to skip already processed data
        fields_map: Field mapping dict, {result_key: item_key}
        track_tokens: Whether to track prompt/completion tokens
        token_fields: Token field mapping dict
        extra_fields_func: Extra fields function (result, item, process_tokens) -> dict
        max_workers: Concurrent thread count, default 1 (sequential execution), recommend 2-8
    """
    all_time = 0
    all_tokens = 0
    all_prompt_tokens = 0
    all_completion_tokens = 0
    failed_items = []

    # Set of processed idxs
    # In material prediction mode, check if material_prediction exists and is not null
    is_material_prediction_mode = config and config.get("material_prediction_only", False)
    if is_material_prediction_mode:
        # Material prediction mode: only keep records where material_prediction field exists and is not null
        processed_idxs = {
            item[idx_field] for item in existing_data
            if item.get("material_prediction") is not None
        } if skip_processed else set()
        failed_count = len([item for item in existing_data if item.get("material_prediction") is None])
        if failed_count > 0 and skip_processed:
            print(f"Found {failed_count} failed material prediction records, will reprocess")
    else:
        processed_idxs = {item[idx_field] for item in existing_data} if skip_processed else set()

    # Filter pending data
    pending_items = []
    for item in data:
        idx = item[idx_field]
        if skip_processed and idx in processed_idxs:
            continue
        pending_items.append(item)
        if max_count and len(pending_items) >= max_count:
            break

    process_count = 0
    total_pending = len(pending_items)

    if total_pending == 0:
        print("No data to process")
        return existing_data

    # Sequential mode (max_workers=1)
    if max_workers <= 1:
        for item in tqdm(pending_items, total=total_pending):
            idx = item[idx_field]
            print(f"---------- Start {idx} --------- ")
            success = False
            result = cost_time = process_tokens = process_cost_time = None

            for attempt in range(max_retries + 1):
                if attempt > 0:
                    print(f"[Retry] idx={idx} attempt {attempt} retry...")
                try:
                    start_time = time.time()
                    result, cost_time, process_tokens = _call_with_timeout(
                        predictor_func, item_timeout,
                        item, model_name=item.get("model_name"), client=client, config=config
                    )
                    end_time = time.time()
                    process_cost_time = end_time - start_time
                    # Check if result is None (indicates skip this sample)
                    if result is None:
                        print(f"[Skip] idx={idx} prediction function returned None, skipping this sample")
                        success = False
                        break
                    if len(result["event"]) > 0:
                        print(f"[RetryOn] idx={idx} result indicates retry needed")
                        if attempt == max_retries:
                            success = False
                        continue
                    success = True
                    break
                except FuturesTimeoutError:
                    print(f"[Timeout] idx={idx} attempt {attempt + 1} timeout (>{item_timeout}s)")
                except Exception as e:
                    print(f"[Error] idx={idx}: {e}")
                    break

            if not success:
                print(f"[Skip] idx={idx} skip")
                failed_items.append(item)
                print()
                continue

            prompt_tokens, completion_tokens = process_tokens
            all_prompt_tokens += prompt_tokens
            all_completion_tokens += completion_tokens
            all_tokens += prompt_tokens + completion_tokens
            all_time += cost_time

            if fields_map:
                for result_key, item_key in fields_map.items():
                    if isinstance(result, dict):
                        item[item_key] = result.get(result_key)

            if token_fields:
                if "cost_time" in token_fields:
                    item[token_fields["cost_time"]] = process_cost_time
                if "prompt_tokens" in token_fields:
                    item[token_fields["prompt_tokens"]] = prompt_tokens
                if "completion_tokens" in token_fields:
                    item[token_fields["completion_tokens"]] = completion_tokens

            if extra_fields_func:
                extra = extra_fields_func(result, item, process_tokens, token_fields=token_fields)
                item.update(extra)

            # Check if already exists (in material prediction mode, update instead of append when reprocessing failed records)
            existing_idx_map = {existing_item[idx_field]: i for i, existing_item in enumerate(existing_data)}
            if idx in existing_idx_map:
                # Update existing record
                existing_data[existing_idx_map[idx]] = item
                print(f"---------- End {idx} (updated existing record) --------- ")
            else:
                # Append new record
                existing_data.append(item)
                print(f"---------- End {idx} --------- ")
            print()
            write_json(save_path, existing_data)
            append_jsonl(save_path.replace(".json", ".jsonl"), item)
            process_count += 1

    else:
        # Concurrent mode
        print(f"Concurrent mode: max_workers={max_workers}, pending={total_pending}")
        lock = threading.Lock()

        def process_one(item):
            idx = item[idx_field]
            print(f"---------- Start {idx} --------- ")
            start_time = time.time()
            try:
                result, cost_time, process_tokens = predictor_func(
                    item, model_name=item.get("model_name"), client=client, config=config
                )
            except Exception as e:
                print(f"---------- Error {idx}: {e} --------- ")
                return idx, None, None, None, None
            end_time = time.time()
            process_cost_time = end_time - start_time

            # If result is None (skip sample) or process_tokens is None
            if result is None:
                return idx, None, None, None, process_cost_time

            prompt_tokens, completion_tokens = process_tokens
            return idx, result, cost_time, process_tokens, process_cost_time

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for item in pending_items:
                future = executor.submit(process_one, item)
                futures[future] = item

            for future in tqdm(as_completed(futures), total=total_pending):
                item = futures[future]
                current_future = future
                success = False
                idx = result = cost_time = process_tokens = process_cost_time = None

                for attempt in range(max_retries + 1):
                    if attempt > 0:
                        print(f"[Retry] idx={item.get(idx_field)} 第 {attempt} 次重试...")
                        current_future = executor.submit(process_one, item)
                    try:
                        idx, result, cost_time, process_tokens, process_cost_time = \
                            current_future.result(timeout=item_timeout)
                        # 检查结果是否为 None（跳过样本）
                        if result is None:
                            print(f"[Skip] idx={item.get(idx_field)} 预测函数返回 None，跳过此样本")
                            success = False
                            break
                        if len(result["event"]) > 0:
                            print(f"[RetryOn] idx={item.get(idx_field)} 结果判定为需重试")
                            if attempt == max_retries:
                                success = False
                            continue
                        success = True
                        break
                    except FuturesTimeoutError:
                        print(f"[Timeout] idx={item.get(idx_field)} 第 {attempt + 1} 次超时（>{item_timeout}s）")
                    except Exception as e:
                        print(f"[Error] idx={item.get(idx_field)}: {e}")
                        break

                if not success or result is None:
                    print(f"[Skip] idx={item.get(idx_field)} 跳过")
                    failed_items.append(item)
                    print()
                    continue

                prompt_tokens, completion_tokens = process_tokens

                with lock:
                    all_prompt_tokens += prompt_tokens
                    all_completion_tokens += completion_tokens
                    all_tokens += prompt_tokens + completion_tokens
                    all_time += cost_time

                    if fields_map:
                        for result_key, item_key in fields_map.items():
                            if isinstance(result, dict):
                                item[item_key] = result.get(result_key)

                    if token_fields:
                        if "cost_time" in token_fields:
                            item[token_fields["cost_time"]] = process_cost_time
                        if "prompt_tokens" in token_fields:
                            item[token_fields["prompt_tokens"]] = prompt_tokens
                        if "completion_tokens" in token_fields:
                            item[token_fields["completion_tokens"]] = completion_tokens

                    if extra_fields_func:
                        extra = extra_fields_func(result, item, process_tokens, token_fields=token_fields)
                        item.update(extra)

                    # Check if already exists (in material prediction mode, update instead of append when reprocessing failed records)
                    existing_idx_map = {existing_item[idx_field]: i for i, existing_item in enumerate(existing_data)}
                    if idx in existing_idx_map:
                        # Update existing record
                        existing_data[existing_idx_map[idx]] = item
                        print(f"---------- End {idx} (updated existing record) --------- ")
                    else:
                        # Append new record
                        existing_data.append(item)
                        print(f"---------- End {idx} --------- ")

                    write_json(save_path, existing_data)
                    append_jsonl(save_path.replace(".json", ".jsonl"), item)
                    process_count += 1
                print()

    # 打印统计
    print(f"一共处理 {process_count} 条数据")
    if process_count > 0:
        print(f"response_time every process: {all_time / process_count}")
        print(f"tokens every process: {all_tokens / process_count}")
        if track_tokens:
            print(f"prompt_tokens every process: {all_prompt_tokens / process_count}")
            print(f"completion_tokens every process: {all_completion_tokens / process_count}")

    # 材质预测模式：保存材质预测结果到指定文件
    if config and config.get("material_prediction_only", False):
        material_prediction_output = config.get("material_prediction_output")

        # 如果没有指定输出路径，默认使用 save_path 所在目录
        if not material_prediction_output:
            save_path_dir = Path(save_path).parent
            material_prediction_output = str(save_path_dir / "material_predictions.json")
            print(f"使用默认材质预测输出路径: {material_prediction_output}")

        if material_prediction_output:
            # 收集所有材质预测结果
            material_predictions = {"query": {}}
            for item in existing_data:
                capture_id = item.get("capture_id", "")
                material_pred = item.get("material_prediction", {})
                if material_pred:
                    chinese_material = material_pred.get("chinese", "其他")
                    material_predictions["query"][capture_id] = chinese_material

            # 保存到文件
            Path(material_prediction_output).parent.mkdir(parents=True, exist_ok=True)
            write_json(material_prediction_output, material_predictions)
            print(f"材质预测结果已保存到: {material_prediction_output}")
            print(f"共预测 {len(material_predictions['query'])} 条材质")

    save_path_table = save_path.replace(".json", ".xlsx")
    excel_processor.json2table(save_path, save_path_table)
    save_path_csv = save_path.replace(".json", ".csv")
    excel_processor.json2table(save_path, save_path_csv, keep_columns=['capture_id', 'pred'])
    fail_path = save_path.replace(".json", "_fail.json")

    if failed_items and fail_path:
        write_json(fail_path, failed_items)
        print(f"重试仍失败 {len(failed_items)} 条，已保存到 {fail_path}")


    return existing_data


# 快速配置版本 - 用于简单场景
def simple_batch_predict(
    data,
    existing_data,
    predict_one_func,
    save_path,
    output_fields,
    idx_field="idx",
):
    """
    简化版批处理 - 假设 predict_one_func 返回的 dict 直接复制到 item

    参数:
        output_fields: 要复制到 item 的字段列表，如 ["classify_result_1st", "description"]
    """
    return batch_predict(
        data=data,
        existing_data=existing_data,
        predictor_func=predict_one_func,
        save_path=save_path,
        idx_field=idx_field,
        fields_map={f: f for f in output_fields},
    )


def batch_predict_from_config(
    args,
    config_path,
    data,
    existing_data,
    save_path,
    client,
    **kwargs
):
    """
    从配置文件加载并执行批量预测

    这是 batch_predict 的配置文件版本，通过 YAML 配置文件指定预测函数和字段映射。

    Args:
        config_path: 配置文件路径
         待预测数据列表
        existing_ 已处理数据列表（用于断点续传）
        save_path: 保存路径
        client: 客户端对象
        **kwargs: 其他参数，会覆盖配置文件中的设置
            - max_workers: 并发线程数
    """
    from .options import load_config
    from .function_loader import load_function_from_config

    # 加载配置
    config = load_config(config_path)
    if not config:
        raise ValueError(f"无法加载配置文件: {config_path}")

    logger.info(f"加载配置文件: {config_path}")
    logger.info(f"配置内容: {config}")

    # 动态加载函数
    predictor_func = load_function_from_config(config)

    # 从配置中提取参数
    batch_kwargs = {
        "idx_field": config.get("idx_field", "idx"),
        "fields_map": config.get("fields_map", {}),
        "track_tokens": config.get("track_tokens", False),
        "max_count": config.get("max_count"),
        "skip_processed": config.get("skip_processed", True),
        "token_fields": config.get("token_fields"),
        "max_workers": config.get("max_workers", 1),
        "item_timeout": config.get("item_timeout", 20),
        "max_retries": config.get("max_retries", 2)
    }

    # 使用 kwargs 覆盖配置（参数优先级更高）
    batch_kwargs.update(kwargs)

    logger.info(f"批处理参数: idx_field={batch_kwargs['idx_field']}, "
                f"track_tokens={batch_kwargs['track_tokens']}, "
                f"max_workers={batch_kwargs['max_workers']}, "
                f"fields_map keys={list(batch_kwargs['fields_map'].keys())}")

    # 调用 batch_predict
    return batch_predict(
        data=data,
        existing_data=existing_data,
        predictor_func=predictor_func,
        config=config,
        save_path=save_path,
        client=client,
        **batch_kwargs
    )


def single_predict_from_config(
    args,
    config_path,
    data,
    existing_data,
    save_path,
    client,
    **kwargs
):
    """
    从配置文件加载并执行单条数据预测

    用于 --single 模式,只处理一条数据(由 args.idx 指定)
    """
    from .options import load_config
    from .function_loader import load_function_from_config

    # 加载配置
    config = load_config(config_path)
    if not config:
        raise ValueError(f"无法加载配置文件: {config_path}")

    logger.info(f"加载配置文件: {config_path}")

    # 动态加载函数
    predictor_func = load_function_from_config(config)

    # 获取要处理的数据索引
    idx_field = config.get("idx_field", "idx")
    target_idx = args.idx

    if target_idx is None:
        raise ValueError("--single 模式需要指定 --idx 参数")

    # 查找目标数据项
    target_item = None
    for item in data:
        if item.get(idx_field) == target_idx:
            target_item = item
            break

    if target_item is None:
        raise ValueError(f"未找到 {idx_field}={target_idx} 的数据项")

    logger.info(f"单条预测模式: 处理 {idx_field}={target_idx}")

    # 从配置中提取参数
    single_kwargs = {
        "idx_field": idx_field,
        "fields_map": config.get("fields_map", {}),
        "track_tokens": config.get("track_tokens", False),
        "token_fields": config.get("token_fields"),
    }

    # 使用 kwargs 覆盖配置
    single_kwargs.update(kwargs)

    # 调用 single_predict
    return single_predict(
        item=target_item,
        existing_data=existing_data,
        predictor_func=predictor_func,
        config=config,
        save_path=save_path,
        client=client,
        **single_kwargs
    )


def single_predict(
    item,
    existing_data,
    predictor_func,
    save_path=None,
    idx_field="idx",
    config=None,
    client=None,
    fields_map=None,
    track_tokens=False,
    token_fields=None,
    extra_fields_func=None,
    print_result=True,
):
    """
    单条数据预测函数

    用于 --single 模式,只处理一条数据
    """
    idx = item[idx_field]
    print(f"\n{'='*60}")
    print(f"  单条预测模式: {idx_field}={idx}")
    print(f"{'='*60}\n")

    # 计时
    start_time = time.time()

    # 调用预测函数
    result, cost_time, process_tokens = predictor_func(
        item, model_name=item.get("model_name"), client=client, config=config
    )

    end_time = time.time()
    process_cost_time = end_time - start_time

    # 处理 tokens
    prompt_tokens, completion_tokens = process_tokens

    # 复制结果字段到 item
    if fields_map:
        for result_key, item_key in fields_map.items():
            if isinstance(result, dict):
                item[item_key] = result.get(result_key)

    # 添加 token 和时间字段
    if token_fields:
        if "cost_time" in token_fields:
            item[token_fields["cost_time"]] = process_cost_time
        if "prompt_tokens" in token_fields:
            item[token_fields["prompt_tokens"]] = prompt_tokens
        if "completion_tokens" in token_fields:
            item[token_fields["completion_tokens"]] = completion_tokens

    # 额外字段处理
    if extra_fields_func:
        extra = extra_fields_func(result, item, process_tokens, token_fields=token_fields)
        item.update(extra)

    # 打印结果
    if print_result:
        print(f"\n{'='*60}")
        print("  预测结果:")
        print(f"{'='*60}")
        if isinstance(result, dict):
            for key, value in result.items():
                value_str = str(value)
                print(f"  {key}: {value_str}")
        print(f"{'='*60}")
        print(f"  处理时间: {process_cost_time:.2f}s")
        print(f"  Prompt tokens: {prompt_tokens}")
        print(f"  Completion tokens: {completion_tokens}")
        print(f"  Total tokens: {prompt_tokens + completion_tokens}")
        print(f"{'='*60}\n")

    # 添加到 existing_data
    existing_data.append(item)

    # 保存(如果指定了保存路径)
    if save_path:
        write_json(save_path, existing_data)
        print(f"✓ 结果已保存到: {save_path}\n")

    return existing_data