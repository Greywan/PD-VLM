import argparse
import os
import yaml

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='./data/yolo_pre_data/collect_poster/0421_0426')
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--file_path','-f', type=str, default=None)
    parser.add_argument('--save_path','-s', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--model','-m', type=str, default=None)
    parser.add_argument('--spe', action='store_true')
    parser.add_argument('--idx', type=int, default=None)
    parser.add_argument("--sheet_name", type=str, default='Sheet1')
    parser.add_argument('--file2','-f2', type=str, default=None)
    parser.add_argument("--group_name", type=str, default='lining')
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--config", type=str, default='./configs/judge_setting.yaml')
    parser.add_argument("--single", action='store_true')

    # Threshold - accuracy plot related parameters
    parser.add_argument("--plot_threshold", action='store_true', help='Plot threshold - accuracy line chart')
    parser.add_argument("--gt_col", type=str, default='图片信息描述是否合理', help='GT column name')
    parser.add_argument("--score_col", type=str, default='judge_combine', help='Score column name')

    return parser.parse_args()


def load_config(config_name=None):
    """
    加载配置文件
    
    Args:
        config_name: 配置文件名称（如 'default.yaml' 或 'config2.yaml'）
                    或完整路径（如 '/path/to/config.yaml'）
                    如果为 None，默认使用 'default.yaml'
    
    Returns:
        dict: 配置字典
    """
    if config_name is None:
        config_name = 'default.yaml'
    
    # 如果提供的是完整路径，直接使用
    if os.path.isabs(config_name) or os.path.exists(config_name):
        config_path = config_name
    else:
        # 否则，从 configs 目录查找
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'configs', config_name)
        if not os.path.exists(config_path):
            print(f"警告: 配置文件 {config_path} 不存在")
            return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    return config_data