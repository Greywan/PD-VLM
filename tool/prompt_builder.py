"""
通用 Prompt 构建器模块
支持从 YAML 配置加载 prompt 模板，并根据参数动态构建最终 prompt
"""

import re
import yaml
from .options import load_config


def extract_placeholders(prompt_template: str) -> list:
    """
    从 prompt 模板中提取所有占位符 {xxx}
    
    Args:
        prompt_template: 包含 {xxx} 占位符的模板字符串
    
    Returns:
        占位符名称列表，如 ['image_text', 'history']
    """
    return re.findall(r'\{(\w+)\}', prompt_template)


def build_prompt(prompt_template: str, **kwargs) -> str:
    """
    根据 prompt 模板和参数构建最终 prompt
    
    Args:
        prompt_template: 包含 {xxx} 占位符的模板字符串
        **kwargs: 占位符对应的参数值
    
    Returns:
        填充后的 prompt 字符串
    
    Raises:
        ValueError: 当缺少必要参数时
    
    Example:
        >>> template = "描述: {image_text}, 评论: {comment_0}"
        >>> build_prompt(template, image_text="图片内容", comment_0="评论内容")
        '描述: 图片内容, 评论: 评论内容'
        
        # 多余参数会被忽略
        >>> build_prompt(template, image_text="图片内容", comment_0="评论内容", history="多余")
        '描述: 图片内容, 评论: 评论内容'
    """
    required = extract_placeholders(prompt_template)
    missing = [p for p in required if p not in kwargs]
    
    if missing:
        raise ValueError(f"缺少必要参数: {missing}，模板需要: {required}")
    
    return prompt_template.format(**kwargs)


if __name__ == "__main__":
    # ========== 单元测试 ==========
    
    # 测试1: 占位符提取
    print("测试1: 占位符提取")
    template = "你好 {name}，你的分数是 {score}"
    placeholders = extract_placeholders(template)
    assert placeholders == ["name", "score"], f"期望 ['name', 'score']，实际 {placeholders}"
    print("  ✅ 通过")
    
    # 测试2: 正常构建
    print("测试2: 正常构建")
    result = build_prompt(template, name="张三", score=95)
    expected = "你好 张三，你的分数是 95"
    assert result == expected, f"期望 '{expected}'，实际 '{result}'"
    print("  ✅ 通过")
    
    # 测试3: 多余参数被忽略
    print("测试3: 多余参数被忽略")
    template_simple = "描述: {image_text}"
    result = build_prompt(template_simple, image_text="图片内容", history="多余参数")
    expected = "描述: 图片内容"
    assert result == expected, f"期望 '{expected}'，实际 '{result}'"
    print("  ✅ 通过")
    
    # 测试4: 参数缺失报错
    print("测试4: 参数缺失报错")
    try:
        build_prompt(template, name="张三")  # 缺少 score
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        assert "缺少必要参数" in str(e), f"错误信息不正确: {e}"
        print(f"  ✅ 通过，错误信息: {e}")
    
    # 测试5: 实际配置文件测试 (如果文件存在)
    print("测试5: 实际配置文件测试")
    config_path = "configs/combine_bestpowerful_descv2.yaml"
    try:
        config = load_config(config_path)
        
        # stage1 测试
        stage1_prompt = build_prompt(config['stage1']['prompt'], image_text="这是一张图片描述")
        print(f"  stage1 prompt 构建成功")
        
        # stage2 测试
        stage2_prompt = build_prompt(config['stage2']['prompt'],
            n_comments=2,
            img_text="这是一张图片描述",
            comment_0="评论内容1",
            comment_1="评论内容2"
        )
        print(f"  stage2 prompt 构建成功")
        print("  ✅ 通过")
        
    except FileNotFoundError:
        print(f"  ⚠️ 配置文件不存在: {config_path}，跳过此测试")
    
    print("\n========== 所有测试通过 ==========")