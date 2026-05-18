
import time
import requests
import dashscope

def call_api_vlm_model(model_name="qwen2-vl-7b-instruct", client=None, base64_img=None, img_format='jpeg', prompt=None, think=False,response_format="text"):
    """
    调用视觉语言模型进行图像目标检测
    
    参数:
    model_name: 模型名称
    img: OpenCV图像对象
    img_format: 图像格式
    prompt: 提示词
    
    返回:
    response: 模型返回的响应内容
    """
    # 将图像编码为base64字符串
    # base64_image = encode_image_from_cv2(img, img_format)
    start_time = time.time()
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": [{"type":"text","text": "You are a helpfull assistant."}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                            # PNG图像：  f"data:image/png;base64,{base64_image}"
                            # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                            # WEBP图像： f"data:image/webp;base64,{base64_image}"
                            "image_url": {"url": f"data:image/{img_format};base64,{base64_img}"},
                            # "image_url": {"url": img},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            # temperature=0,
            # top_p=0.3,
            max_tokens=2048,
            extra_body={"enable_thinking":think},
            response_format={"type": response_format}
        )
    except Exception as e:
        print(e)
        response = "模型调用失败"
        return response, 0, (0, 0)
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    print(f"prompt_tokens: {prompt_tokens}")
    print(f"completion_tokens: {completion_tokens}")
    # if all_tokens is not None:
    #     all_tokens += prompt_tokens
    response = completion.choices[0].message.content
    end_time = time.time()
    cost_time = end_time - start_time
    print(f"模型响应时间: {cost_time}秒")

    return response, cost_time, (prompt_tokens, completion_tokens)


def call_api_vlm_model_messages_dashscope(model_name="qwen2-vl-7b-instruct", api_key=None, messages=None, think=False,response_format="text"):
    """
    调用语言模型
    
    参数:
    model_name: 模型名称
    messages: 消息列表
    
    返回:
    response: 模型返回的响应内容
    """
    # 将图像编码为base64字符串
    # base64_image = encode_image_from_cv2(img, img_format)
    start_time = time.time()
    try:
        response = dashscope.MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key=api_key,
        model=model_name, # 此处以qwen-vl-max为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=messages,
        max_tokens=2048,
        extra_body={"enable_thinking":think},
        response_format={"type": response_format}
        )
    except Exception as e:
        print(e)
        response = "模型调用失败"
        return response, 0, (0, 0)
    print(response)
    prompt_tokens = response.usage.input_tokens
    completion_tokens = response.usage.output_tokens
    print(f"prompt_tokens: {prompt_tokens}")
    print(f"completion_tokens: {completion_tokens}")
    # if all_tokens is not None:
    #     all_tokens += prompt_tokens
    response = response.output.choices[0].message.content[0]["text"]
    end_time = time.time()
    cost_time = end_time - start_time
    print(f"模型响应时间: {cost_time}秒")

    return response, cost_time, (prompt_tokens, completion_tokens)


def call_api_lm_model_messages(model_name="qwen2-vl-7b-instruct", client=None, messages=None, think=False,response_format="text", temperature=None):
    """
    调用语言模型

    参数:
    model_name: 模型名称
    messages: 消息列表
    temperature: 采样温度 (None=使用API默认值, 0=确定性, 1=高随机性)

    返回:
    response: 模型返回的响应内容
    """
    # 将图像编码为base64字符串
    # base64_image = encode_image_from_cv2(img, img_format)
    start_time = time.time()
    try:
        kwargs = dict(
            model=model_name,
            messages=messages,
            max_tokens=2048,
            extra_body={"enable_thinking":think},
            response_format={"type": response_format}
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        completion = client.chat.completions.create(**kwargs)
    except Exception as e:
        print(e)
        response = "模型调用失败"
        return response, 0, (0, 0)
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    print(f"prompt_tokens: {prompt_tokens}")
    print(f"completion_tokens: {completion_tokens}")
    # if all_tokens is not None:
    #     all_tokens += prompt_tokens
    response = completion.choices[0].message.content
    end_time = time.time()
    cost_time = end_time - start_time
    print(f"模型响应时间: {cost_time}秒")

    return response, cost_time, (prompt_tokens, completion_tokens)


def call_api_llm_model(model_name="qwen2-vl-7b-instruct", client=None, prompt=None, all_time=None, all_tokens=None):
    """
    调用语言模型
    
    参数:
    model_name: 模型名称
    prompt: 提示词
    
    返回:
    response: 模型返回的响应内容
    """
    # 将图像编码为base64字符串
    # base64_image = encode_image_from_cv2(img, img_format)
    start_time = time.time()
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": [{"type":"text","text": prompt}]},
            ],
            temperature=0.01,
            # top_p=0.3,
            max_tokens=500,
        )
    except Exception as e:
        print(e)
        response = "模型调用失败"
        return response, 0, 0
    end_time = time.time()
    print(f"模型响应时间: {end_time - start_time}秒")
    prompt_tokens = completion.usage.prompt_tokens
    print(f"prompt_tokens: {prompt_tokens}")
    if all_tokens is not None:
        all_tokens += prompt_tokens
    response = completion.choices[0].message.content
    if all_time is not None:
        all_time += (end_time - start_time)
    return response, all_time, all_tokens


def call_service_vlm_model(service_url="http://127.0.0.1:1080/hmi/eco_pic_desc", request_data=None, result_map=None):
    """
    调用视觉语言模型进行图像目标检测
    
    参数:
    model_name: 模型名称
    img: OpenCV图像对象
    img_format: 图像格式
    prompt: 提示词
    
    返回:
    response: 模型返回的响应内容
    """
    # 将图像编码为base64字符串
    # base64_image = encode_image_from_cv2(img, img_format)

    start_time = time.time()
    try:
        response = requests.post(
        url=service_url, # 预发链接: https://af-model-interaction-pre.antgroup-inc.cn/hmi/eco_pic_desc
        json=request_data,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    except Exception as e:
        print(e)
        response = "模型调用失败"
        result_map["eventSubTypeAnno"] = '模型调用失败'
        return result_map, 0, (0,0)
    prompt_tokens = 0
    print(f"prompt_tokens: {prompt_tokens}")
    response = response.json().get("resultMap", {})
    end_time = time.time()
    cost_time = end_time - start_time
    print(f"模型响应时间: {cost_time}秒")
    return response, cost_time, (prompt_tokens, 0)