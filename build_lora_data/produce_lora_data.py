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
                print(f"{Colors.BLUE}💭 思考链: {sample_think}{Colors.END}")
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
    作为化工安全专家，请严格按DeepSeek的思考链格式回答：

    【回答格式】
    <think>
    1. 识别关键因素
    2. 分析事故演化路径
    3. 引用相关法规标准
    4. 选择处置方案的依据
    </think>
    
    【回答要求】
    ✦ 实际回答应包含但不限于以下内容：
    - 工程控制措施（含设备参数）
    - 管理控制措施（如适用）
    - 应急响应步骤
    - 验证指标和方法
    
    示例：
    问题：液氨储罐泄漏应如何处置？
    
    <think>
    1. 物质特性：液氨（CAS 7664-41-7），沸点-33.34℃，蒸气密度0.6
    2. 主要风险：急性中毒（IDLH 300ppm）、燃爆风险（16-25%）
    3. 法规依据：GB 50351-2014第4.2.3条
    4. 处置原则：根据泄漏量分级响应
    5. 其他思考过程视情况而定
    </think>
    
    处置方案：
    1. 工程措施：
       - 启动ESD-2000紧急切断系统（响应时间＜2秒）
       - 开启D9型消防水幕（流量60L/s，覆盖率≥95%）
    2. 管理措施（根据现场情况）：
       - 视情况实施双人巡检制度
       - 必要时进行作业许可管理
    3. 应急流程：
       - 5分钟内建立300m警戒区（使用MK-4型检测仪）
       - 30分钟完成倒罐操作（流速≤3m/s）
    4. 验证标准：
       - 优先保证环境检测达标（PGM-7600＜30ppm）
       - 其他验证措施视情况而定
    5.其他内容：
       - 其它生成内容视具体情况而定
"""

    # 文件配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    question_file = os.path.join(base_dir, "questions_set.py")
    output_file = os.path.join(base_dir, "chemical_safety_deepseek.json")
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