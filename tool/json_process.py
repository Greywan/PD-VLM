"""
JSON data processing tool

Usage:
    # Class-based approach
    from tool.json_process import JsonProcessor
    processor = JsonProcessor()
    processor.load("input.json").filter_failed_responses().save("output.json")

    # Command line approach
    python -m tool.json_process -f input.json -s output.json --add_idx
"""
import os
import sys
import random
import json
import ast
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool.utils import process_string_remove_background



def read_json(file_path):
    """Read JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(file_path, data):
    """Write JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def write_jsonl(file_path, data):
    """Write JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def append_jsonl(file_path, data):
    """Append JSONL file"""
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
def json_to_jsonl(data, file_path):
    """Write JSON array to JSONL file line by line

    Args:
        data: JSON data list, e.g. [{"a": 1}, {"b": 2}]
        file_path: Output JSONL file path
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


class JsonProcessor:
    """JSON data processor"""

    def __init__(self, data=None):
        self.data = data or []

    # ==================== Basic read/write ====================

    @staticmethod
    def loads(data_str):
        """Parse JSON string, supports json and ast formats"""
        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(data_str)
            except (ValueError, SyntaxError):
                return None

    # ==================== Chaining operations ====================

    def load(self, file_path):
        """Load from file"""
        self.data = read_json(file_path)
        return self

    def save(self, file_path):
        """Save to file"""
        write_json(file_path, self.data)
        return self

    def filter_failed_responses(self, key='new_response'):
        """Filter failed responses"""
        failed_values = {"解析失败", "图片数据异常", "模型调用失败", "无法解析模型输出"}
        print(f"Start filtering failed responses, key: {key}")
        print(f"Original data {len(self.data)} items")
        self.data = [item for item in self.data if item.get(key) not in failed_values]
        print(f"Remaining {len(self.data)} items after filtering")
        return self

    def filter_by_key_value(self, key_name, value):
        """Filter by key-value"""
        self.data = [item for item in self.data if item.get(key_name) == value]
        return self

    def exclude_by_idx(self, exclude_ids):
        """Exclude specified idx"""
        self.data = [item for item in self.data if item.get("idx") not in exclude_ids]
        return self

    def include_by_idx(self, exclude_ids):
        """Include specified idx"""
        self.data = [item for item in self.data if item.get("idx") in exclude_ids]
        return self
    
    def add_index(self, idx_key='idx'):
        """Add sequential index"""
        for i, item in enumerate(self.data):
            item[idx_key] = i
        return self

    def remove_background(self, key):
        """Remove background text"""
        for item in self.data:
            if key in item:
                item[key] = process_string_remove_background(item[key])
        return self

    def merge_fields(self, fields_map):
        """Merge fields, copy from source field to target field"""
        for item in self.data:
            for src_key, dst_key in fields_map.items():
                item[dst_key] = item.get(src_key)
        return self

    def sample(self, nums):
        """Random sampling"""
        actual_nums = min(nums, len(self.data))
        self.data = random.sample(self.data, actual_nums)
        return self

    def keep_keys(self, keys):
        """Only keep specified keys, delete other fields

        Args:
            keys: List of keys to keep, e.g. ['a', 'b', 'c', 'd']

        Returns:
            self, supports chaining
        """
        self.data = [{k: item.get(k) for k in keys} for item in self.data]
        return self

    def sort_by_key(self, key, reverse=False):
        """Sort data by specified key

        Args:
            key: Field name to sort by, e.g. 'idx'
            reverse: Whether to sort in descending order, default False (ascending)

        Returns:
            self, supports chaining
        """
        self.data.sort(key=lambda item: item.get(key, 0), reverse=reverse)
        return self

# ==================== Command line interface ====================

def process_command_line():
    """Process command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(description='JSON data processing tool')

    # Basic parameters
    parser.add_argument('-f', '--file_path', type=str, help='Input JSON file path')
    parser.add_argument('-s', '--save_path', type=str, help='Output JSON file path')
    parser.add_argument('-f2', '--file2', type=str, help='Second JSON file path')
    parser.add_argument('--key', type=str, default='new_response', help='Field name')
    parser.add_argument('--value', type=str, help='Field value')
    parser.add_argument('--exclude_ids', type=str, help='Excluded idx, comma separated')
    parser.add_argument('--nums', type=int, default=100, help='Random sampling count')

    # Feature selection
    parser.add_argument('--add_idx', action='store_true', help='Add sequential index')
    parser.add_argument('--filter_fail', action='store_true', help='Filter failed responses')
    parser.add_argument('--filter_by_key', action='store_true', help='Filter by key-value')
    parser.add_argument('--filter_by_key_value', action='store_true', help='Filter by key-value')
    parser.add_argument('--exclude', action='store_true', help='Exclude specified idx')
    parser.add_argument('--include', action='store_true', help='Include specified idx')
    parser.add_argument('--merge', action='store_true', help='Merge two JSON files')
    parser.add_argument('--sample', action='store_true', help='Random sampling')
    parser.add_argument('--remove_bg', action='store_true', help='Remove background text')
    parser.add_argument('--keep_keys', type=str, help='Keys to keep, comma separated, e.g.: a,b,c,d')
    parser.add_argument('--sort_by', type=str, help='Sort by specified key, e.g.: idx')
    parser.add_argument('--sort_reverse', action='store_true', help='Descending order (default ascending)')
    parser.add_argument('--to_jsonl', action='store_true', help='Convert JSON to JSONL format')

    return parser.parse_args()


def main():
    args = process_command_line()

    if args.file_path is None:
        print("错误：必须提供输入文件路径 (-f/--file_path)")
        print("可用功能：")
        print("  --add_idx        添加顺序索引")
        print("  --filter_fail    过滤失败响应")
        print("  --filter_by_key_value  按key-value过滤")
        print("  --filter_by_key  按key过滤")
        print("  --exclude        排除指定idx")
        print("  --merge          合并两个JSON文件")
        print("  --sample         随机采样")
        print("  --remove_bg      移除背景文字")
        print("  --sort_by        按指定key排序")
        print("  --to_jsonl       将JSON转换为JSONL格式")
        sys.exit(1)

    if args.save_path is None:
        print("错误：必须提供输出文件路径 (-s/--save_path)")
        sys.exit(1)

    processor = JsonProcessor()

    try:
        # 根据不同功能执行
        if args.add_idx:
            processor.load(args.file_path).add_index().save(args.save_path)
            print(f"完成：已添加索引并保存到 {args.save_path}")

        elif args.filter_fail:
            key = args.key or 'new_response'
            processor.load(args.file_path).filter_failed_responses(key).save(args.save_path)
            print(f"完成：已过滤失败响应并保存到 {args.save_path}")

        elif args.filter_by_key:
            if not args.key or not args.value:
                print("错误：--filter_by_key 需要 --key 和 --value 参数")
                sys.exit(1)
            processor.load(args.file_path).filter_by_key(args.key, args.value).save(args.save_path)
            print(f"完成：已按 {args.key}={args.value} 过滤并保存到 {args.save_path}")

        elif args.exclude:
            if not args.exclude_ids:
                print("错误：--exclude 需要 --exclude_ids 参数")
                sys.exit(1)
            exclude_ids = [int(x.strip()) for x in args.exclude_ids.split(',')]
            processor.load(args.file_path).exclude_by_idx(exclude_ids).save(args.save_path)
            print(f"完成：已排除idx {exclude_ids} 并保存到 {args.save_path}")
        elif args.include:
            if not args.exclude_ids:
                print("错误：--exclude 需要 --exclude_ids 参数")
                sys.exit(1)
            exclude_ids = [int(x.strip()) for x in args.exclude_ids.split(',')]
            processor.load(args.file_path).include_by_idx(exclude_ids).save(args.save_path)
            print(f"完成：已排除idx {exclude_ids} 并保存到 {args.save_path}")
        elif args.merge:
            if not args.file2:
                print("错误：--merge 需要提供第二个文件 (-f2/--file2)")
                sys.exit(1)
            data1 = read_json(args.file_path)
            print(f"读取第一个文件：{len(data1)} 条数据")
            data2 = read_json(args.file2)
            print(f"读取第二个文件：{len(data2)} 条数据")
            merge_data = data1 + data2
            print(f"合并后：{len(merge_data)} 条数据")
            write_json(args.save_path, merge_data)
            print(f"完成：已合并两个文件并保存到 {args.save_path}")

        elif args.sample:
            nums = args.nums or 100
            processor.load(args.file_path).sample(nums).save(args.save_path)
            print(f"完成：已随机采样 {nums} 条并保存到 {args.save_path}")

        elif args.remove_bg:
            if not args.key:
                print("错误：--remove_bg 需要 --key 参数")
                sys.exit(1)
            processor.load(args.file_path).remove_background(args.key).save(args.save_path)
            print(f"完成：已移除 {args.key} 的背景文字并保存到 {args.save_path}")
        elif args.keep_keys:
            keys = [k.strip() for k in args.keep_keys.split(',')]
            processor.load(args.file_path).keep_keys(keys).save(args.save_path)
            print(f"完成：已保留指定key并保存到 {args.save_path}")
        elif args.sort_by:
            processor.load(args.file_path).sort_by_key(args.sort_by, reverse=args.sort_reverse).save(args.save_path)
            order = "降序" if args.sort_reverse else "升序"
            print(f"完成：已按 {args.sort_by} {order}排序并保存到 {args.save_path}")
        elif args.to_jsonl:
            data = read_json(args.file_path)
            json_to_jsonl(data, args.save_path)
            print(f"完成：已转换为JSONL并保存到 {args.save_path}")
        else:
            print("请指定要执行的功能，使用 --help 查看可用功能")

    except Exception as e:
        print(f"错误：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()