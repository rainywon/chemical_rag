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
    作为化工安全专家，请按照以下DeepSeek思考链格式回答化工安全问题：

    【思考链要求】
    <think>
    请选择以下一种分析方法进行思考，但不要在回答中提及"方案A/B/C"或复制括号内的说明文字：

    方法一：物质与设备分析路径
    1. 危险化学品特性分析
    2. 工艺设备条件评估
    3. 暴露情景与后果分析
    4. 法规标准查询
    5. 控制措施等级评估

    方法二：事故与应急分析路径
    1. 事故类型判断
    2. 事故发展路径分析
    3. 应急资源评估
    4. 应急处置优先级判断
    5. 措施有效性评估

    方法三：系统与管理分析路径
    1. 安全管理体系要素识别
    2. 防护层分析
    3. 人因工程考量
    4. 安全文化影响因素
    5. 管理措施有效性评估
    </think>

    【注意】思考时参考上述框架，但在写出<think>内容时，请:
    1. 不要包含"方法一/二/三"字样
    2. 不要包含括号内的解释说明
    3. 直接给出实质性分析内容

    【回答内容要求】
    ✦ 专业深度要求：
    - 危化品信息：提供CAS编号、关键危险参数(LEL/UEL、IDLH等)
    - 设备参数：明确型号、参数规格(如流量、压力、材质)
    - 法规引用：具体到条款号、发布年份、适用范围
    - 定量数据：给出精确数值(如安全距离、检测限值、操作参数)

    ✦ 回答必须包含(适用项)：
    1. 工程控制措施：
       - 主动控制系统(如联锁、自动停车系统、抑爆系统)
       - 被动防护设施(如防火堤、防爆墙、泄压装置)
       - 监测预警设备(如气体检测仪、温度监控)

    2. 管理控制措施：
       - 操作规程要求(如巡检频率、记录要求)
       - 培训内容与频次
       - 审批与许可制度
       - 应急演练要求

    3. 应急响应流程：
       - 分级响应条件与标准
       - 时间节点要求(如1分钟内报警、5分钟内疏散)
       - 指挥体系与职责分工
       - 外部支援与协调

    4. 验证与评估方法：
       - 效果验证指标(如浓度降至LEL以下)
       - 检测方法与设备(如红外探测器、便携式检测仪)
       - 恢复条件与验收标准

    ✦ 表达多样性要求：
    - 使用决策树形式表达条件判断
    - 采用分级或分类方式组织内容
    - 包含清晰的时间顺序与优先级
    - 使用专业术语准确表达概念

    【回答示例】
    问题：液氨储罐泄漏应如何处置？

    <think>
    1. 物质特性：液氨（CAS 7664-41-7），沸点-33.34℃，蒸气密度0.6，LC50(大鼠,1h)=4230ppm
    2. 主要风险：急性毒性(IDLH 300ppm)、低温灼伤、可燃范围(16-25%)
    3. 法规依据：GB 50351-2014《储运液氨》第4.2.3条、GB/T 32127《氨气泄漏事故应急处置技术导则》
    4. 事故发展：小泄漏→蒸发扩散→形成毒气云团→人员中毒；大泄漏→快速气化→压力升高→设备破裂→爆炸
    5. 控制措施选择依据：隔离源、疏散人员、控制/稀释泄漏、监测浓度
    </think>

    处置方案：

    一、初始评估与响应（0-5分钟）
    1. 工程控制：
       - 立即启动ESD-2000紧急切断系统（响应时间<2秒）隔离泄漏源
       - 开启D9型水幕系统（流量60L/s，覆盖率≥95%）抑制氨气扩散
       - 关闭非防爆电器，切断火源（泄漏区域10m范围内）

    2. 疏散与警戒：
       - 建立警戒区：小泄漏（<50kg）→100m；大泄漏（>50kg）→300m
       - 人员疏散：上风向撤离，垂直风向疏散（不应顺风或逆风）

    二、控制与消除（5-30分钟）
    1. 泄漏源控制：
       - 小泄漏（<50kg）：使用V型密封工具（耐低温-50℃）堵漏
       - 大泄漏（>50kg）：实施液氨倒罐（流速≤3m/s，背压≤0.2MPa）

    2. 监测与评估：
       - 使用MK-4型氨气检测仪（量程0-1000ppm，精度±2%）进行边界监测
       - 每5分钟测量一次下风向100m、300m、500m处浓度
       
    三、应急恢复条件（验证指标）
    1. 泄漏源完全隔离且泄漏停止
    2. 警戒区边界氨气浓度<25ppm（TLV-TWA）持续15分钟
    3. 设备压力稳定在正常工作范围（0.6-0.8MPa）30分钟以上
    4. 周边水体pH值7-8.5（无明显氨污染）

    注意：若检测到警戒区边界氨气浓度>300ppm（IDLH），应立即扩大警戒范围并申请上级支援。
"""

    # 文件配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    question_file = os.path.join(base_dir, "questions.py")
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