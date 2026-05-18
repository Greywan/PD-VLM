import re
import json
import cv2
import base64
import os
from PIL import Image
import requests
from urllib.parse import urlparse

def count_digits_regex(text):
    """使用正则表达式统计数字字符个数"""
    if not text:
        return 0
    # 匹配所有数字字符（0-9）
    digits = re.findall(r'\d', text)
    return len(digits)


def convert_to_description(response):
    try:
        response_str = response.replace("\n", "").replace("\t", "").replace("```json", "").replace("```", "")
        if response_str[-2:] == "}}":
            response_str = response_str[:-2] + "}"
        # 解析JSON字符串
        response_data = json.loads(response_str)
        description = response_data.get("description", "")
        description = process_string_remove_background(description)
    except:
        description = "解析失败"
    # results.append(description)
    result = {
        "description": description,
        "thought": "",
    }
    return result

def convert_to_description_2(response):
    try:
        description = ""  # 初始化变量
        thought = ""      # 初始化变量
        dialogue_above_intent = ""

        result_match = re.search(r'result:\s*(\{.*?})', response)
        if result_match:
            results = json.loads(result_match.group(1))
            description = results.get("图片描述", "")
            description = process_string_remove_background(description)
        else:
            description = "解析失败"
        thought_match = re.search(r'Thought:(.*?)(?:\nresult:|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            thought = ""
    except:
        description = "解析失败"
        thought = ""
        dialogue_above_intent = "解析失败"
    
    result = {
        "description": description,
        "thought": thought,
        "dialogue_above_intent": dialogue_above_intent
    }
    return result

def convert_to_description_3(response):
    """
    新的处理函数，用于提取思考过程和用户意图
    
    参数:
    response: 包含"### 思考过程"和"### 结果"的响应文本
    
    返回:
    result: 包含thought和user_intent的字典
    """
    try:
        thought = ""      # 初始化思考过程变量
        description = ""  # 初始化用户意图变量
        dialogue_above_intent = ""
        # 提取思考过程部分
        thought_match = re.search(r'### 思考过程\s*\n(.*?)(?=\n### 结果|\n---|\Z)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            thought = ""

        # 提取结果部分的JSON数据
        result_match = re.search(r'### 结果\s*\n```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if result_match:
            try:
                results = json.loads(result_match.group(1))
                description = results.get("图片描述", "")
                description = process_string_remove_background(description)
            except json.JSONDecodeError:
                description = "解析失败"
        else:
            description = "解析失败"
            
    except Exception as e:
        thought = "解析失败"
        description = "解析失败"
        dialogue_above_intent = "解析失败"

    result = {
        "description": description,
        "thought": thought,
        "dialogue_above_intent": dialogue_above_intent
    }
    return result

def convert_to_description_4(response):
    try:
        description = ""  # 初始化变量
        thought = ""      # 初始化变量
        dialogue_above_intent = ""

        result_match = re.search(r'result:\s*(\{.*?})', response)
        if result_match:
            results = json.loads(result_match.group(1))
            description = results.get("图片描述", "")
            description = process_string_remove_background(description)
            dialogue_above_intent = results.get("用户意图", "")
        else:
            description = "解析失败"
            thought = "解析失败"
        # thought_match = re.search(r'Thought:(.*?)(?:\nresult:|$)', response, re.DOTALL)
        # if thought_match:
        #     thought = thought_match.group(1).strip()
        # else:
        #     thought = ""
    except:
        description = "解析失败"
        thought = ""
    
    result = {
        "description": description,
        "thought": thought,
        "dialogue_above_intent": dialogue_above_intent
    }
    return result

def convert_to_intent_description(response):
    try:
        description = ""  # 初始化变量
        thought = ""      # 初始化变量
        dialogue_above_intent = ""

        result_match = re.search(r'result:\s*(\{.*?})', response)
        results = convert_str2list(response)
        if result_match:
            results = json.loads(result_match.group(1))
            description = results.get("图片描述", "")
            description = process_string_remove_background(description)
            dialogue_above_intent = results.get("用户意图", "")
            classify_result = results.get("图片分类", "")
        elif len(results) > 0:
            description = results.get("图片描述", "")
            dialogue_above_intent = results.get("用户意图", "")
            classify_result = results.get("图片分类", "")
        else:
            description = "解析失败"
            thought = "解析失败"
            classify_result = "解析失败"
        # thought_match = re.search(r'Thought:(.*?)(?:\nresult:|$)', response, re.DOTALL)
        # if thought_match:
        #     thought = thought_match.group(1).strip()
        # else:
        #     thought = ""
    except:
        description = "解析失败"
        thought = ""
        classify_result = ""
    
    result = {
        "description": description,
        "thought": thought,
        "dialogue_above_intent": dialogue_above_intent,
        "classify_result": classify_result
    }
    return result

def convert_to_classify2st_description(response):
    try:
        description = ""  # 初始化变量
        thought = ""      # 初始化变量
        dialogue_above_intent = ""

        result_match = re.search(r'result:\s*(\{.*?})', response)
        results = convert_str2list(response)
        if result_match:
            results = json.loads(result_match.group(1))
            description = results.get("图片描述", "")
            description = process_string_remove_background(description)
            dialogue_above_intent = results.get("用户意图", "")
            classify_result_1st = results.get("图片一级分类", "")
            classify_result_2nd = results.get("图片二级分类", "")
        elif len(results) > 0:
            description = results.get("图片描述", "")
            dialogue_above_intent = results.get("用户意图", "")
            classify_result_1st = results.get("图片一级分类", "")
            classify_result_2nd = results.get("图片二级分类", "")
            img_quality = results.get("图片质量", "")
        else:
            description = "解析失败"
            thought = "解析失败"
            classify_result_1st = "解析失败"
            classify_result_2nd = "解析失败"
            img_quality = "解析失败"
        # thought_match = re.search(r'Thought:(.*?)(?:\nresult:|$)', response, re.DOTALL)
        # if thought_match:
        #     thought = thought_match.group(1).strip()
        # else:
        #     thought = ""
    except:
        description = "解析失败"
        thought = "解析失败"
        classify_result_1st = "解析失败"
        classify_result_2nd = "解析失败"
        img_quality = "解析失败"

    result = {
        "description": description,
        "thought": thought,
        "dialogue_above_intent": dialogue_above_intent,
        "classify_result_1st": classify_result_1st,
        "classify_result_2nd": classify_result_2nd,
        "img_quality": img_quality
    }
    return result


def convert_to_jitu_description(response):
    try:
        order_number = ""  # 订单编号
        item_description = ""      # 物品描述
        response_str = response.replace("\n", "").replace("\t", "").replace("```json", "").replace("```", "")
        try:
            results = json.loads(response_str)
            result_match = True
            order_number = results.get("order_number", "")
            item_description = results.get("item_description", "")
        except:
            result_match = False
            order_number = ""
            item_description = ""            
        # thought_match = re.search(r'Thought:(.*?)(?:\nresult:|$)', response, re.DOTALL)
        # if thought_match:
        #     thought = thought_match.group(1).strip()
        # else:
        #     thought = ""
    except:
        order_number = ""
        item_description = ""
    
    result = {
        "order_number": order_number,
        "item_description": item_description,
    }
    return result


def convert_to_shoe_description(response):
    try:
        judge_res = ""  # 订单编号
        problem_type = ""      # 物品描述
        location = ""
        confidence = ""
        response_str = response.replace("\n", "").replace("\t", "").replace("```json", "").replace("```", "")
        try:
            results = json.loads(response_str)
            result_match = True
            judge_res = results.get("检测结果", "")
            problem_type = results.get("问题类型", "")
            location = results.get("位置描述", "")
            confidence = results.get("严重程度", "")
        except:
            result_match = False          
        # thought_match = re.search(r'Thought:(.*?)(?:\nresult:|$)', response, re.DOTALL)
        # if thought_match:
        #     thought = thought_match.group(1).strip()
        # else:
        #     thought = ""
    except:
        result_match = False          
   
    result = {
        "judge_res": judge_res,
        "problem_type": problem_type,
        "location": location,
        "confidence": confidence,
    }
    return result


def convert_str2list(response):
    """安全解析模型响应"""
    try:
        # 方法1：尝试提取 ```json...``` 代码块中的内容
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # 方法2：尝试提取第一个完整的 JSON 对象
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(response[start:end+1])
        
        response_str = response.replace("\n", "").replace("\t", "").replace("```json", "").replace("```", "")
        return json.loads(response_str)
    except:
        return []

def convert_str2dict(response):
    """安全解析模型响应"""
    try:
        # 方法1：尝试提取 ```json...``` 代码块中的内容
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # 方法2：尝试提取第一个完整的 JSON 对象
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(response[start:end+1])
        
        response_str = response.replace("\n", "").replace("\t", "").replace("```json", "").replace("```", "")
        return json.loads(response_str)
    except:
        return None


def extract_defect_and_explain(response_text):
    """从非标准 JSON 文本中用正则提取 defect 和 explain 字段"""
    result = {}
    # 提取 defect 值（数字）
    defect_match = re.search(r'"defect"\s*:\s*(\d+(?:\.\d+)?)', response_text)
    if defect_match:
        result["defect"] = float(defect_match.group(1))
    # 提取 explain 值（字符串）
    explain_match = re.search(r'"explain"\s*:\s*"((?:[^"\\]|\\.)*)"', response_text)
    if explain_match:
        result["explain"] = explain_match.group(1)
    return result if result else None
    
def parsed_jitu_description(result, img):
    if len(result['order_number']) <= 14 or not result['order_number'].startswith('JT'):
        result['order_number'] = ''
    if result['order_number'] == '' and result['item_description'] == '':
        return img
    if result['item_description'] == '':
        return result['order_number']
    if result['order_number'] == '':
        return result['item_description']
    else: return result['order_number'] + ',' + result['item_description']


def parsed_station_description(results, img):
    all_order_message = ""
    try:
        if len(results) > 0 :
            for idx, result in enumerate(results):
                single_order_message = f"运单{idx}:"
                if result['company'] != "":
                    single_order_message += f"{result['company']} "
                if result['tracking_number'] !=  "":
                    if len(result['tracking_number']) > 9: 
                        single_order_message += f"{result['tracking_number']} "
                if result['pickup_code'] != "":
                    single_order_message += f"取件码: {result['pickup_code']} "
                single_order_message += ";"
                all_order_message += single_order_message
            return all_order_message
        else: return img
    except:
        return img

def process_string_remove_background(text):
    # 使用逗号和句号作为分隔符分割字符串
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if char in ["。"]:
            sentences.append(current_sentence)
            current_sentence = ""
    
    # 如果最后一个句子没有结束符，也添加到列表中
    if current_sentence:
        sentences.append(current_sentence)
    
    # 过滤掉包含"背景"的句子
    filtered_sentences = [s for s in sentences if "背景" not in s]
    
    # 重新组合字符串
    result = "".join(filtered_sentences)
    
    return result

def encode_image_from_cv2(cv2_img, img_format='jpg'):
    """
    将cv2图像对象编码为base64字符串
    
    参数:
    cv2_img: OpenCV图像对象
    img_format: 图像格式，默认为'jpg'
    
    返回:
    base64_string: 编码后的base64字符串
    """
    # 将图像编码为指定格式的内存缓冲区
    success, buffer = cv2.imencode(f'.{img_format}', cv2_img)
    if not success:
        return None
    
    # 将缓冲区转换为bytes，然后编码为base64字符串
    bytes_data = buffer.tobytes()
    base64_string = base64.b64encode(bytes_data).decode('utf-8')
    
    return base64_string

def read_img_fromurl(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'}
    invalid_extensions = {'.gif', '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.pdf', '.doc', '.txt', '.zip', '.rar', '.html'}
    try:
        # url 扩展名预检查
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        if any(path.endswith(ext) for ext in invalid_extensions):
            print(f"URL扩展名检查失败，不支持的文件类型: {url}")
            return None
        response = requests.get(url, stream=True, headers=headers, timeout=30)
        # response.raise_for_status()  # 检查HTTP状态码
        image_obj = Image.open(response.raw)
        return image_obj
    except requests.exceptions.ConnectionError as e:
        print(f"网络连接错误，无法访问图片URL: {url}")
        print(f"错误详情: {e}")
        return None
    except requests.exceptions.Timeout as e:
        print(f"请求超时，无法访问图片URL: {url}")
        print(f"错误详情: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP错误，无法访问图片URL: {url}")
        print(f"错误详情: {e}")
        return None
    except Exception as e:
        print(f"读取图片时发生未知错误，URL: {url}")
        print(f"错误详情: {e}")
        return None

def check_args(raw, new):
    if new:
        raw = new
    return raw

def extract_tag_content(text, tag_name, default=None):
    """
    从文本中提取指定 XML 标签内的内容

    参数:
        text: 包含标签的原始文本
        tag_name: 标签名称（如 'answer'）
        default: 未找到标签时返回的默认值，默认为 None

    返回:
        标签内的内容（字符串），未找到时返回 default

    示例:
        >>> extract_tag_content('<answer>1.0</answer>', 'answer')
        '1.0'
        >>> extract_tag_content('无标签文本', 'answer')
        None
    """
    if not text or not tag_name:
        return default
    pattern = rf'<{tag_name}>(.*?)</{tag_name}>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return default

def extract_defect_and_explain(response_text):
    """从非标准 JSON 文本中提取 defect 和 explain"""
    result = {}
    
    # 提取 defect 值（数字）
    defect_match = re.search(r'"defect"\s*:\s*(\d+(?:\.\d+)?)', response_text)
    if defect_match:
        result["defect"] = float(defect_match.group(1))
    
    # 提取 explain 值（字符串，贪婪匹配到最后一个闭合引号前）
    explain_match = re.search(r'"explain"\s*:\s*"((?:[^"\\]|\\.)*)"', response_text)
    if explain_match:
        result["explain"] = explain_match.group(1)
    
    return result if result else None


if __name__ == "__main__":
    url = "https://dd-static.jd.com/ddimgp/jfs/t20270303/390267/3/9432/43543/697853f0F2fbdaa90/09362d021c2b959b.jpg"
    img = read_img_fromurl(url)
    print(img)