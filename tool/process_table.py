import pandas as pd
import json
import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool.json_process import read_json
from tool.options import get_args
from tool.utils import read_img_fromurl


def extract_url_from_text(text):
    text = text.replace('https://', 'http://')
    """从文本中提取第一个URL"""
    # 匹配http/https开头的URL，直到遇到非URL字符
    url_pattern = r'http?://[^\s\u4e00-\u9fff]+'
    match = re.search(url_pattern, text)
    return match.group(0) if match else None


def save_json_to_excel(file_path, excel_path):
    data = read_json(file_path)

    # # 从每个记录中删除idx字段
    # for item in data:
    #     if 'idx' in item:
    #         del item['idx']

    df = pd.DataFrame(data)
    df.to_excel(excel_path, index=False)

def combine_excel_files(file_a, file_b, save_path):
    # 读取两个 Excel 文件
    df1 = pd.read_excel(file_a)
    df2 = pd.read_excel(file_b)
    # 合并数据
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df.to_excel(save_path, index=False)

def update_excel_file(excel_file, new_excel_file, key_name, add_key_name):
    """
    更新 Excel 文件，添加 add_key_name 列，根据原始 Excel key_name 对应列和一定规则填充值

    参数:
    excel_file (str): 原始 Excel 文件路径
    new_excel_file (str): 更新后的 Excel 文件路径
    """
    # 读取原始 Excel 文件
    df = pd.read_excel(excel_file, dtype={'chat_record_id': str})

    # 新建 'gt' 列，根据条件赋值
    def assign_gt(row):
        if row[key_name] == 0:
            return row['pred_check0']
        elif row[key_name] == 1:
            return row['pred_check1']
        elif row[key_name] == 2:
            return row['pred_check2']
        else:
            return row['pred']  # 或者其他默认值
    
    def assign_device(row):
        if '前置' in row[key_name]:
            return 'pc'
        else :
            return 'exter'
    
    def judge_img_url(row):
        img_url = row[key_name]
        img_url = extract_url_from_text(img_url)
        img = read_img_fromurl(img_url)
        if img is None:
            return 0
        else:
            return 1
        
    df[add_key_name] = df.apply(judge_img_url, axis=1)
    # 保存到新的 Excel 文件
    df.to_excel(new_excel_file, index=False)

def clean_data_for_json(data):
    """清洗数据以便转换为JSON格式"""
    if isinstance(data, dict):
        return {k: clean_data_for_json(v) for k, v in data.items() if pd.notna(v)}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data if pd.notna(item)]
    elif pd.isna(data):
        return None
    else:
        return data

def convert_xlsx_to_json(xlsx_path, json_path, sheet_name=None, orient='records'):
    """
    将Excel文件转换为JSON文件
    
    参数:
    - xlsx_path: Excel文件路径
    - json_path: 输出JSON文件路径
    - sheet_name: 工作表名称，默认为第一个工作表
    - orient: JSON格式方向，可选 'records', 'index', 'columns', 'values'
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(xlsx_path):
            raise FileNotFoundError(f"Excel文件不存在: {xlsx_path}")
        
        # 读取Excel文件
        df = pd.read_excel(xlsx_path)
        
        # 清洗数据
        # df = df.where(pd.notnull(df), None)
        
        # 转换为字典格式
        if orient == 'records':
            data = df.to_dict('records')
        elif orient == 'index':
            data = df.to_dict('index')
        elif orient == 'columns':
            data = df.to_dict()
        elif orient == 'values':
            data = df.values.tolist()
        else:
            data = df.to_dict('records')
        
        # 清洗数据
        cleaned_data = clean_data_for_json(data)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        # 保存为JSON文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        return True
        
    except Exception as e:
        print(f"转换失败: {str(e)}")
        return False


def table2list(file_path, sheet_name=None, orient='records'):
    try:
        # 检查输入文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel文件不存在: {file_path}")
        
        # 读取Excel文件
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)

        
        # 清洗数据
        # df = df.where(pd.notnull(df), None)
        
        # 转换为字典格式
        if orient == 'records':
            data = df.to_dict('records')
        elif orient == 'index':
            data = df.to_dict('index')
        elif orient == 'columns':
            data = df.to_dict()
        elif orient == 'values':
            data = df.values.tolist()
        else:
            data = df.to_dict('records')
        
        # 清洗数据
        cleaned_data = clean_data_for_json(data)
        return cleaned_data
    except Exception as e:
        print(f"转换失败: {str(e)}")
        return None


if __name__ == "__main__":
    args = get_args()
    file_path = './data/bpo/.xlsx'
    save_path = './data/test/1010_1688_http_img.xlsx'
    file_path = args.file_path
    save_path = args.save_path
    
    # save_json_to_excel(file_path, save_path)
    # combine_excel_files(file_path, save_path, save_path)
    # update_excel_file(file_path, save_path, 'chat_context', 'is_img')
    # convert_xlsx_to_json(file_path, save_path)