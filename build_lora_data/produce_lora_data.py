import json
import os
import re
import time
import concurrent.futures
from functools import partial
from zhipuai import ZhipuAI

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

class CotValidator:
    @staticmethod
    def validate(answer):
        """验证CoT格式的核心要素"""
        # 检查思考标签
        if not re.search(r'<think>.*?</think>', answer, re.DOTALL):
            raise ValueError("缺少<think>思考标签")
        
        # 检查实际回答内容
        if answer.count('<think>') != answer.count('</think>'):
            raise ValueError("思考标签不匹配")
        
        # 提取实际回答部分
        actual_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        if not actual_answer:
            raise ValueError("实际回答内容为空")
        return True

def load_questions(py_path):
    """从Python文件加载问题列表"""
    try:
        with open(py_path, "r", encoding="utf-8") as f:
            namespace = {}
            exec(f.read(), namespace)
            return namespace.get("questions", [])
    except Exception as e:
        raise RuntimeError(f"解析问题文件失败: {str(e)}")

def load_existing_data(json_path):
    """加载已有数据并建立问题索引"""
    processed = set()
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)  # 直接加载JSON数组
                for entry in data:
                    processed.add(entry["instruction"].strip())
        except Exception as e:
            os.rename(json_path, f"{json_path}.bak")
            print(f"{Colors.YELLOW}⚠ 数据文件损坏，已备份: {str(e)}{Colors.END}")
    return processed

def generate_deepseek_entry(question, answer):
    """生成DeepSeek格式数据条目"""
    return {
        "instruction": question,
        "input": "",
        "output": answer
    }


def save_with_backup(data, path):
    """带备份的安全保存（JSON数组格式）"""
    temp_path = f"{path}.tmp"
    try:
        # 读取现有数据
        existing = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)

        # 合并数据
        combined = existing + data

        # 写入临时文件
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)

        # 原子替换
        if os.path.exists(path):
            os.replace(path, f"{path}.bak")
        os.rename(temp_path, path)
    except Exception as e:
        print(f"{Colors.RED}保存失败: {str(e)}{Colors.END}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_question(client, system_prompt, question, error_log, retry=3):
    """带重试机制的思考链处理"""
    for attempt in range(retry):
        try:
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            answer = response.choices[0].message.content
            
            # 格式标准化处理
            answer = re.sub(r'(\d+)\s*(米|m)\b', r'\1m', answer)
            answer = re.sub(r'(?i)<think>', '<think>', answer)
            answer = re.sub(r'(?i)</think>', '</think>', answer)
            answer = re.sub(r'第\s*(\d+)\s*条', r'第\1条', answer)
            
            # 验证思考链格式
            CotValidator.validate(answer)
            
            return generate_deepseek_entry(question, answer)
            
        except Exception as e:
            if attempt < retry - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"{Colors.YELLOW}⚠ 第{attempt+1}次重试，等待{wait_time}秒...{Colors.END}")
                time.sleep(wait_time)
            else:
                with open(error_log, "a", encoding="utf-8") as f:
                    f.write(f"{question}|||{str(e)}\n")
                return None

def process_question_wrapper(client, system_prompt, error_log, question):
    """带状态提示的包装函数"""
    try:
        print(f"{Colors.BLUE}🟡 开始处理: {question[:30]}...{Colors.END}")
        result = process_question(client, system_prompt, question, error_log)
        if result:
            print(f"{Colors.GREEN}✅ 成功处理: {question[:30]}...{Colors.END}")
            # 打印思考链示例
            think_content = re.search(r'<think>(.*?)</think>', result["output"], re.DOTALL)
            if think_content:
                sample_think = think_content.group(1)[:50].replace('\n', ' ') + "..."
        else:
            print(f"{Colors.YELLOW}⚠️ 空响应: {question[:30]}...{Colors.END}")
        return result
    except Exception as e:
        print(f"{Colors.RED}❌ 处理失败: {question[:30]}... 错误: {str(e)}{Colors.END}")
        return None

def process_batch(client, system_prompt, error_log, batch):
    """带统计的批次处理"""
    print(f"\n{Colors.BLUE}▶ 开始批次处理 ({len(batch)}个问题) {Colors.END}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        process_fn = partial(process_question_wrapper, client, system_prompt, error_log)
        results = list(executor.map(process_fn, batch))

    success = sum(1 for r in results if r)
    failed = len(results) - success
    print(f"{Colors.GREEN}✔ 成功: {success} {Colors.YELLOW}⚠ 失败: {failed}{Colors.END}")
    return [r for r in results if r]

def main():
    client = ZhipuAI(api_key="4e0779dc66414dc4afe0872680957d40.HnKsmRuaJjYQHEUL")
    
    system_prompt = """
    作为化工安全专家，你需要回答化工行业一线工作人员提出的实际问题。请注意以下几点：

    【用户问题特点】
    - 问题通常来自实际工作场景
    - 可能包含紧急情况处理需求
    - 可能有不精确或不完整的表述
    - 可能混合多个相关问题

    【回答要求】
    1. 思考过程（<think>标签内）：
       - 自由分析问题的核心需求和关键点
       - 考虑问题背后的安全隐患和潜在风险
       - 结合相关法规标准、规范和最佳实践
       - 思考可能的解决方案和注意事项

    2. 实际回答：
       - 直接回应用户的核心问题
       - 给出清晰、具体、可操作的建议
       - 强调安全注意事项和关键步骤
       - 适当补充必要的专业知识
       - 回答语气应专业但友好，避免过于学术化

    【回答风格】
    - 开门见山，先解决核心问题
    - 使用简洁明了的语言
    - 适当使用行业术语但注意解释
    - 按操作步骤或重要性排序信息
    - 关注实用性和可操作性

    【回答示例】
    问题：加氢装置催化剂床层温度突然升高，该怎么处理？

    <think>
    1. 情况分析：
       - 催化剂床层温度突然升高是加氢装置常见的危险情况
       - 可能原因：进料组成变化、氢气流量不足、催化剂中毒或烧结
       - 风险：温度失控可能导致反应器超温、催化剂烧损、甚至引发爆炸
       - 关键点：需要快速判断原因并采取措施降温

    2. 处理思路：
       - 立即检查：进料组成、流量、氢气纯度、循环气量
       - 应急降温措施评估
       - 是否需要装置降负荷或紧急停车
       - 后续催化剂评估和更换计划
    </think>

    床层温度突然升高需要立即处理，按以下步骤操作：

    1. 紧急措施：
       - 增加循环氢气量，提高氢油比
       - 降低进料量减少热负荷
       - 必要时启动紧急注氮系统冷却

    2. 原因快速排查：
       - 检查进料组成（是否有高烯烃或含氧物质）
       - 确认循环氢压力和纯度
       - 检查床层压降是否异常

    3. 决策点：
       - 如温度持续上升超过设计值30℃以上，准备紧急停车
       - 如压降急剧增加，可能是催化剂结焦，需立即降负荷

    4. 后续处理：
       - 记录完整工艺参数用于分析
       - 安排催化剂活性检测
       - 制定催化剂再生或更换计划

    记住：床层温度失控是加氢装置最危险的情况之一，宁可装置降负荷也不要冒险运行。
"""

    

    # 文件配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    question_file = os.path.join(base_dir, "random_12000_questions.py")
    output_file = os.path.join(base_dir, "chemical_safety_deepseek_2.json")
    error_log = os.path.join(base_dir, "deepseek_errors.log")

    # 加载数据
    processed = load_existing_data(output_file)
    all_questions = load_questions(question_file)
    todo_questions = [q for q in all_questions if q not in processed]
    
    print(f"{Colors.BLUE}📊 待处理问题：{len(todo_questions)}/{len(all_questions)}{Colors.END}")

    # 分批处理
    batch_size = 200  # 减小批次保证质量
    for idx in range(0, len(todo_questions), batch_size):
        batch = todo_questions[idx:idx+batch_size]
        print(f"\n{Colors.BLUE}🔷 处理批次 {idx//batch_size + 1} [数量：{len(batch)}]{Colors.END}")
        
        results = process_batch(client, system_prompt, error_log, batch)
        
        if results:
            save_with_backup(results, output_file)
            print(f"{Colors.GREEN}✅ 已保存{len(results)}条数据{Colors.END}")

if __name__ == "__main__":
    main()