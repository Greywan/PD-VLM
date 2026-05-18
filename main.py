import os
from tool.options import get_args, load_config
import os
from openai import OpenAI
from datetime import datetime
from loguru import logger
import shutil

from tool.utils import *
from tool.batch_utils import batch_predict_from_config, single_predict_from_config
from tool.process_table import table2list
from tool.json_process import read_json

def main():
    args = get_args()

    data_path = "./data/bpo/labeled/class_false_up.json"
    data_path = check_args(data_path, args.file_path)

    if data_path.endswith(".xlsx"):
        data = table2list(data_path)
    else:
        data = read_json(data_path)

    output_dir = "./outputs/"
    output_dir = check_args(output_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    config_path = "./configs/business_functions/bpo_predict.yaml"
    config_path = check_args(config_path, args.config)
    config = load_config(config_path)

    if output_dir:
        log_name = f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
        log_path = os.path.join(output_dir, f"{log_name}.log")
        logger.add(log_path)

    logger.info(f"Start judge batch predict with config_path: {args.config}")
    logger.info(config)
    # model_name = "qwen-vl-max"
    # model_name = "qwen2.5-vl-72b-instruct"
    model_name = "qwen3-vl-flash"
    model_name = check_args(model_name, args.model)


    save_path = './data/business_data/0701/output/apipre_qwenvl25_72_w8_choose_chat_pm21_img2048.json'
    save_path = check_args(save_path, args.save_path)
    save_path_dir = os.path.dirname(save_path)
    os.makedirs(save_path_dir, exist_ok=True)
    if os.path.exists(os.path.join(save_path_dir, os.path.basename(config_path))):
        logger.info(f"Config file already exists in {save_path_dir}")
    else:
        shutil.copy(config_path, save_path_dir)
    # Read existing file data first (if file exists)
    try:
        existing_data = read_json(save_path) if os.path.exists(save_path) else []
    except:
        existing_data = []

    if None == config.get("DASHSCOPE_API_KEY"):
        api_key = os.getenv('DASHSCOPE_API_KEY')
    else:
        api_key = config.get("DASHSCOPE_API_KEY")

    base_url = config.get("DASHSCOPE_BASE_URL")

    client = OpenAI(
        # If environment variable is not configured, replace the following line with: api_key="sk-xxx"
        api_key=api_key,
        base_url=base_url,
    )

    if args.single:
        single_predict_from_config(
            args=args,
            config_path=config_path,
            data=data,
            existing_data=existing_data,
            save_path=save_path,
            client=client
        )
    else:
        batch_predict_from_config(
            args=args,
            config_path=config_path,
            data=data,
            existing_data=existing_data,
            save_path=save_path,
            client=client
        )

if __name__ == '__main__':
    main()