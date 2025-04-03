import json
import os
from sklearn.model_selection import train_test_split

# 定义文件名
# filename = "chemical_safety_deepseek.json"
filename = "chemical_safety_deepseek.json"

# 检查文件是否存在
if not os.path.exists(filename):
    print(f"文件 {filename} 不存在")
else:
    # 打开并读取文件
    with open(filename, "r", encoding="utf-8") as file:
        # 加载 JSON 数据
        data = json.load(file)

        # 检查数据是否为列表或字典
        if isinstance(data, (list, dict)):
            # 将数据转换为列表（如果是字典，取其键值对）
            if isinstance(data, dict):
                data = list(data.items())

            # 按照 8:2 的比例分割数据
            train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

            # 将训练集和测试集保存到 JSON 文件
            with open("train_data.json", "w", encoding="utf-8") as train_file:
                json.dump(train_data, train_file, ensure_ascii=False, indent=4)

            with open("test_data.json", "w", encoding="utf-8") as test_file:
                json.dump(test_data, test_file, ensure_ascii=False, indent=4)

            print("数据已成功分割并保存到 train_data.json 和 test_data.json 文件中")
        else:
            print("JSON 数据不是一个列表或字典，无法分割")