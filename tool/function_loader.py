"""
函数动态加载工具
用于从配置文件中动态加载预测函数
"""
import importlib
import logging

logger = logging.getLogger(__name__)


def load_function_from_config(config):
    """
    根据配置动态加载函数

    Args:
        config: 配置字典,必须包含:
            - predictor_func: 函数名称
            - predictor_module: 函数所在的模块路径

    Returns:
        function: 加载的函数对象

    Raises:
        ValueError: 配置缺失必要字段
        ImportError: 模块导入失败
        AttributeError: 函数不存在

    Example:
        config = {
            "predictor_func": "bpo_predict_allonemodel_eco_fortrain",
            "predictor_module": "imgvqa_function.eco_img_desc"
        }
        func = load_function_from_config(config)
        result = func(data_item, model_name, output_dir, client)
    """
    func_name = config.get("predictor_func")
    module_name = config.get("predictor_module")

    if not func_name:
        raise ValueError("配置文件必须包含 'predictor_func' 字段")
    if not module_name:
        raise ValueError("配置文件必须包含 'predictor_module' 字段")

    try:
        # 动态导入模块
        module = importlib.import_module(module_name)

        # 获取函数对象
        if not hasattr(module, func_name):
            raise AttributeError(f"模块 '{module_name}' 中不存在函数 '{func_name}'")

        func = getattr(module, func_name)

        if not callable(func):
            raise ValueError(f"'{func_name}' 不是可调用的函数")

        logger.info(f"成功加载函数: {module_name}.{func_name}")
        return func

    except ImportError as e:
        logger.error(f"导入模块失败: {module_name}, 错误: {e}")
        raise
    except Exception as e:
        logger.error(f"加载函数失败: {module_name}.{func_name}, 错误: {e}")
        raise


def validate_function_signature(func, expected_params=None):
    """
    验证函数签名是否符合预期

    Args:
        func: 要验证的函数
        expected_params: 期望的参数名列表,如果为None则不验证

    Returns:
        bool: 是否通过验证
    """
    import inspect

    if not callable(func):
        return False

    if expected_params is None:
        return True

    try:
        sig = inspect.signature(func)
        actual_params = list(sig.parameters.keys())

        # 检查期望的参数是否都存在
        for param in expected_params:
            if param not in actual_params:
                logger.warning(f"函数缺少期望的参数: {param}")
                return False

        return True
    except Exception as e:
        logger.error(f"验证函数签名时出错: {e}")
        return False