import base64
import os
from loguru import logger
from PIL import Image
from tool.utils import read_img_fromurl, check_args
from tool.cv_tools import resize_img
import cv2
import numpy as np
from tool.options import get_args

def encode_image(image_path):
    """
    将图像文件编码为 Base64 字符串

    :param image_path: 图像文件的路径
    :return: Base64 编码的字符串
    """
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_encoded_data = base64.b64encode(image_data)
        base64_image = base64_encoded_data.decode("utf-8")
        return base64_image


def return_img_format(url, img_format='jpeg'):
    if 'png' in url:
        read_img_format = 'PNG'
        img_format = 'png'
    else:
        read_img_format = 'JPEG'
        img_format = 'jpeg'
    return read_img_format, img_format


def return_imgcontent_warning(url):
    success = False
    img_content = None
    
    if url.startswith('https:') or url.startswith('http:'):
        img_content = {"type": "image_url", "image_url": {"url": f"{url}"}}
        success = True
    elif os.path.exists(url):
        # 如果是本地文件路径，读取并转换为Base64
        read_img_format, img_format = return_img_format(url)
        base64_image = encode_image(url)
        img_content = {"type": "image_url", "image_url": {"url": f"data:image/{img_format};base64,{base64_image}"}}
        success = True
    else:
        logger.info('get_response invalid image path:{}'.format(url))
    return success, img_content


def read_img(img_path):
    if os.path.exists(img_path):
        image_obj = Image.open(img_path)
    else:
        image_obj = read_img_fromurl(img_path)
    return image_obj

def read_img_returncontent(img_path, max_size=6000):
    if os.path.exists(img_path):
        image_obj = Image.open(img_path)
    else:
        image_obj = read_img_fromurl(img_path)
    if image_obj is None:
        return None, [], None
    read_img_format, img_format = return_img_format(img_path)
    base64_img = resize_img(image_obj, max_size=max_size, format=read_img_format)
    img_content = {"type": "image_url", "image_url": {"url": f"data:image/{img_format};base64,{base64_img}"}}

    return True, img_content, image_obj

def read_img_returnimgobj(img_path, max_size=6000, min_size=224):
    if os.path.exists(img_path):
        image_obj = Image.open(img_path)
    else:
        image_obj = read_img_fromurl(img_path)
    # 获取原始尺寸
    width, height = image_obj.size
    is_too_small = width < min_size or height < min_size
    if is_too_small:
        logger.info(f"图片尺寸过小: {img_path} ({width}x{height}) 小于阈值 {min_size}x{min_size}")
    if image_obj is None:
        return None, "", None
    read_img_format, img_format = return_img_format(img_path)
    base64_img, resized_image = resize_img(image_obj, max_size=max_size, format=read_img_format)
    size_info = {
        "new_width": resized_image.width,
        "new_height": resized_image.height,
        "is_too_small": is_too_small
    }
    return resized_image, img_format, base64_img, size_info


def get_base64_img(img_path, max_size=2048):
    if os.path.exists(img_path):
        image_obj = Image.open(img_path)
    else:
        image_obj = read_img_fromurl(img_path)
    if image_obj is None:
        return "图片数据异常", ""
    read_img_format, img_format = return_img_format(img_path)
    # 将图像编码为base64字符串
    base64_img = resize_img(image_obj, max_size=max_size, format=read_img_format)
    return base64_img, img_format


def is_white_background(img, threshold=50):
    # 转换到HSV空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 定义白色范围（可根据实际情况调整）
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    
    # 创建掩码
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 计算白色区域占比
    white_ratio = np.sum(mask) / (mask.shape[0] * mask.shape[1])
    return white_ratio
    

def is_white_background_cv2(img, threshold_percent=0.2):
    """计算图片白色区域占比
    Args:
        img: cv2.imread读取的图片对象数组格式
    Returns:
        float: 白色区域占比(0.0-1.0)
    """
    # 1. 加载图片
    # img = cv2.imread(image_path)
    if img is None:
        return False
    
    # 2. 转换到 HSV 空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 定义“白色”的范围 (亮度高，饱和度低)
    # V > 220 (亮度足够高), S < 30 (颜色足够淡)
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([180, 30, 255])
    
    # 创建掩膜
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 3. 计算白色占比
    white_pixels = cv2.countNonZero(mask)
    total_pixels = img.shape[0] * img.shape[1]
    white_ratio = white_pixels / total_pixels
    
    # 4. 辅助检查：四角纯度检测 (取每个角 5% 的区域)
    # h, w = img.shape[:2]
    # cp = 0.05 # corner percent
    # corners = [
    #     img[0:int(h*cp), 0:int(w*cp)],      # 左上
    #     img[0:int(h*cp), int(w*(1-cp)):w],  # 右上
    #     img[int(h*(1-cp)):h, 0:int(w*cp)],  # 左下
    #     img[int(h*(1-cp)):h, int(w*(1-cp)):w] # 右下
    # ]
    
    # corner_variance = np.mean([np.var(c) for c in corners])
    # corner_brightness = np.mean([np.mean(c) for c in corners])

    # 逻辑判定：
    # - 白色占比要高
    # - 四角方差要低 (说明背景平滑，没有杂物)
    # - 四角亮度要极高
    # if white_ratio > threshold_percent and corner_variance < 50 and corner_brightness > 230:
    return white_ratio
    # if white_ratio > threshold_percent:
    #     return True, white_ratio
    # else:
    #     return False, white_ratio
    
def get_white_ratio(img):
    """计算图片白色区域占比
    Args:
        img: PIL.Image对象
    Returns:
        float: 白色区域占比(0.0-1.0)
    """
    # 核心实现：3行代码完成计算
    arr = np.array(img)
    white_mask = (arr[:,:,:3] >= 255).all(axis=2)
    return white_mask.sum() / white_mask.size

if __name__ == '__main__':
    img_path = '/Users/liujc/Downloads/1734006000000.jpg'
    args = get_args()
    img_path = check_args(img_path, args.file_path)

    img = cv2.imread(img_path)
    print(is_white_background_cv2(img))
