import base64
import os
from io import BytesIO
from PIL import Image
import re

def resize_img(image_obj, max_size=2048, format="JPEG"):
    """
    按最长边进行等比例缩放图片
    Args:
        image_obj: PIL Image对象
        max_size: 最长边的最大尺寸，默认2048
    Returns:
        base64编码的图片字符串
    """
    # 获取原始尺寸
    width, height = image_obj.size
    
    # 计算缩放比例
    if width > height:
        # 宽度是长边
        if width > max_size:
            scale_ratio = max_size / width
            new_width = max_size
            new_height = int(height * scale_ratio)
        else:
            new_width, new_height = width, height
    else:
        # 高度是长边
        if height > max_size:
            scale_ratio = max_size / height
            new_height = max_size
            new_width = int(width * scale_ratio)
        else:
            new_width, new_height = width, height
    print(f"原始尺寸: {width}x{height}, 缩放后尺寸: {new_width}x{new_height}")
    # 进行缩放
    resized_image = image_obj.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 转换为base64
    buffered = BytesIO()
    resized_image.save(buffered, format=format, quality=95)
    buffered.seek(0)
    base64_image = base64.b64encode(buffered.read()).decode('utf-8')
    
    return base64_image, resized_image

def base64_to_image(base64_string, save_path, filename=None):
    """
    将base64编码的图片解码并保存为图片文件
    
    Args:
        base64_string (str): base64编码的图片字符串
        save_path (str): 保存图片的目录路径
        filename (str, optional): 保存的文件名，如果不提供则自动生成
    
    Returns:
        str: 保存的完整文件路径
    
    Raises:
        ValueError: 当base64字符串无效时
        IOError: 当文件保存失败时
    """
    try:
        # 处理可能包含数据URL前缀的base64字符串
        # 例如: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
        if base64_string.startswith('data:'):
            # 提取base64数据部分
            base64_data = base64_string.split(',', 1)[1]
            # 提取图片格式
            format_match = re.search(r'data:image/(\w+);base64,', base64_string)
            image_format = format_match.group(1).upper() if format_match else 'JPEG'
        else:
            base64_data = base64_string
            image_format = 'JPEG'  # 默认格式
        
        # 解码base64数据
        image_data = base64.b64decode(base64_data)
        
        # 使用BytesIO创建内存中的图片对象
        image_buffer = BytesIO(image_data)
        image = Image.open(image_buffer)
        
        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 生成文件名
        if filename is None:
            # 根据图片格式生成默认文件名
            extension = image_format.lower()
            if extension == 'jpeg':
                extension = 'jpg'
            filename = f"decoded_image.{extension}"
        
        # 构建完整的文件路径
        full_path = os.path.join(save_path, filename)
        
        # 保存图片
        # 如果是RGBA模式但要保存为JPEG，需要转换为RGB
        if image.mode == 'RGBA' and image_format.upper() == 'JPEG':
            # 创建白色背景
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1])  # 使用alpha通道作为mask
            rgb_image.save(full_path, format=image_format, quality=95)
        else:
            image.save(full_path, format=image_format, quality=95)
        
        print(f"图片已成功保存到: {full_path}")
        return full_path
        
    except Exception as e:
        raise IOError(f"保存图片失败: {str(e)}")


def base64_to_image_with_auto_format(base64_string, save_path, filename_prefix="decoded_image"):
    """
    将base64编码的图片解码并保存，自动检测图片格式
    
    Args:
        base64_string (str): base64编码的图片字符串
        save_path (str): 保存图片的目录路径
        filename_prefix (str): 文件名前缀
    
    Returns:
        str: 保存的完整文件路径
    """
    try:
        # 处理base64字符串
        if base64_string.startswith('data:'):
            base64_data = base64_string.split(',', 1)[1]
        else:
            base64_data = base64_string
        
        # 解码base64数据
        image_data = base64.b64decode(base64_data)
        
        # 使用BytesIO创建内存中的图片对象
        image_buffer = BytesIO(image_data)
        image = Image.open(image_buffer)
        
        # 自动检测图片格式
        image_format = image.format if image.format else 'JPEG'
        
        # 确定文件扩展名
        extension_map = {
            'JPEG': 'jpg',
            'PNG': 'png',
            'GIF': 'gif',
            'BMP': 'bmp',
            'WEBP': 'webp'
        }
        extension = extension_map.get(image_format, 'jpg')
        
        # 生成文件名
        filename = f"{filename_prefix}.{extension}"
        
        # 调用主函数保存图片
        return base64_to_image(base64_string, save_path, filename)
        
    except Exception as e:
        raise IOError(f"自动格式保存失败: {str(e)}")


def batch_base64_to_images(base64_list, save_path, filename_prefix="image"):
    """
    批量将base64编码的图片解码并保存
    
    Args:
        base64_list (list): base64编码的图片字符串列表
        save_path (str): 保存图片的目录路径
        filename_prefix (str): 文件名前缀
    
    Returns:
        list: 保存的文件路径列表
    """
    saved_paths = []
    
    for i, base64_string in enumerate(base64_list):
        try:
            filename = f"{filename_prefix}_{i+1:03d}"
            saved_path = base64_to_image_with_auto_format(base64_string, save_path, filename)
            saved_paths.append(saved_path)
        except Exception as e:
            print(f"保存第 {i+1} 张图片失败: {str(e)}")
            saved_paths.append(None)
    
    return saved_paths


# 使用示例
if __name__ == "__main__":
    # 示例用法
    sample_base64 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."  # 这里应该是完整的base64字符串
    
    try:
        # 单张图片保存
        saved_path = base64_to_image(sample_base64, "./output", "test_image.jpg")
        print(f"图片保存成功: {saved_path}")
        
        # 自动格式检测保存
        auto_saved_path = base64_to_image_with_auto_format(sample_base64, "./output", "auto_image")
        print(f"自动格式图片保存成功: {auto_saved_path}")
        
    except Exception as e:
        print(f"保存失败: {e}")
