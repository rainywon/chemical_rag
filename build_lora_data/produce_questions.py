import threading
import requests
import json
import random
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import hashlib

# === 全局配置 ===
API_CONFIG = {
    "ENDPOINT": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "KEY": "4e0779dc66414dc4afe0872680957d40.HnKsmRuaJjYQHEUL",
    "MAX_TOKENS": 70,
    "TEMP": 0.6,
    "BATCH_SIZE": 30,
    "OUTPUT_FILE": "questions_set.py",
    "CONCURRENCY": 12,  # 根据CPU核心数优化
    "TOPICS": [
        ("化学品全生命周期管理", [
            "MSDS合规性动态更新", "禁忌物料智能识别系统", "相容性矩阵数字化管理",
            "微量分装抑爆措施", "过期化学品自催化风险评估", "剧毒物虹膜双鉴系统",
            "纳米材料工程控制措施", "自反应物质绝热存储", "临界量动态监测算法",
            "管输化学品段塞流控制", "电化学腐蚀抑制剂选择", "聚合抑制剂在线监测",
            "间歇工艺残留物互反应", "废弃溶剂闪点监测", "阻燃剂环境毒性评估"
        ]),

        ("化工数字化安全", [
            "DCS系统共模故障预防", "SIS安全仪表时滞补偿", "PLC梯形图逻辑漏洞",
            "工业防火墙过程协议解析", "操作日志时序异常检测", "数字孪生机理模型验证",
            "APC参数漂移锁定", "本安型无线仪表组网", "AI特征重要性可信验证",
            "MES配方防误传机制", "批量控制序列容错", "SDG模型动态更新",
            "报警风暴根源分析", "OPC UA证书轮换", "工艺知识图谱冲突检测"
        ]),

        ("高危工艺安全控制", [
            "硝化工艺猝冷剂注入速率", "氯化反应自由基淬灭", "加氢催化剂床层飞温",
            "氧化过程热点迁移监测", "电解槽膜电极电位均衡", "聚合反应凝胶效应控制",
            "磺化过程发烟酸浓度", "胺化过程PH振荡抑制", "氟化反应材料相容性",
            "微反应器通道堵塞预警", "超临界流体泄压结晶", "流化床静电累积监测",
            "固定床局部烧结检测", "反应精馏共沸物控制", "微波加热热点消除"
        ]),

        ("跨国化工合规", [
            "REACH暴露场景本地化", "GHS混合物分类逻辑", "OSHA过程安全要素衔接",
            "ATEX区域划分动态调整", "HAZMAT多式联运规则", "BAT废水预处理标准",
            "PSM机械完整性衔接", "RMP最坏情景模拟", "TSCA重要新用途申报",
            "中国GB与欧盟CLP协调", "跨境应急物资储备", "宗教区防爆等级适配"
        ]),

        ("化工本质安全设计", [
            "最小化原则的HAZOP应用", "替代溶剂的共沸点验证", "缓和措施的能量阈值",
            "简化流程的HAZID分析", "故障安全型仪表气源", "冗余系统的共因失效",
            "人机界面Fitts定律应用", "三维可达性仿真验证", "防误操作的力矩限制",
            "模块化设计的泄漏包容", "颜色编码的色弱适配", "泄压面积动态计算"
        ]),

        ("化工园区安全", [
            "多米诺效应概率阈值", "公共管廊腐蚀监测点", "封闭化周界气体监测",
            "企业互审的HAZOP交叉", "应急资源覆盖模型", "危废焚烧二噁英控制",
            "智能巡检路径优化", "入园物质配伍矩阵", "重大源视频智能分析",
            "地沟油气云监测", "暴雨内涝水力模型", "社区风险沟通沙盘"
        ]),

        ("化工变更安全", [
            "MOC电子签批追溯", "临时变更时限控制", "同类事故情景匹配",
            "变更影响矩阵权重", "承包商黑名单共享", "PSSR检查项动态生成",
            "回退方案的RTO验证", "文档版本冲突检测", "VR沉浸式变更培训",
            "工艺参数变更裕度", "设备材质变更兼容", "仪表量程变更验证"
        ]),

        ("化工极端工况应对", [
            "台风工况锚固力计算", "极寒伴热系统冗余", "洪涝防浮筒失效",
            "地震加速度报警阈值", "限电工况安全停车", "疫情封闭最低配置",
            "防恐周界微波探测", "战时原料替代方案", "APT攻击工艺防御",
            "雷击浪涌保护", "沙尘暴密封强化", "干旱消防水源保障"
        ]),

        ("化工安全文化", [
            "安全观察STOP卡优化", "未遂事件根本原因库", "安全里程碑指标",
            "行为安全ABC分析", "虚拟现实事故体验", "家属安全告知程序",
            "领导力安全足迹", "心智模式双回路", "交接班安全仪式",
            "新人安全认知地图", "承包商文化融入", "安全故事知识图谱"
        ]),

        ("化工实验室安全", [
            "微量放热反应量热", "高压釜磁耦合监测", "手套箱氧含量控制",
            "放射性同位素台账", "生物安全柜气流", "激光器互锁装置",
            "XRD样品制备规范", "动物实验3R原则", "基因编辑物理防护",
            "超临界色谱安全", "微波消解压力突变", "自燃物质筛分"
        ]),

        ("化工新能源安全", [
            "储氢材料氢脆系数", "液氢BOG回收系统", "氨裂解催化剂中毒",
            "CO2管道减压相变", "甲醇重整积碳控制", "固态电池热失控",
            "飞轮轴承失效监测", "核能制氢氚防护", "光伏清洗PID效应",
            "锂电电解液阻燃", "生物柴油冷滤点", "氢燃料电池反极"
        ]),

        ("化工退役安全", [
            "残留物拉曼光谱识别", "管线吹扫RBI技术", "反向拆除应力分析",
            "污染区三维建模", "爆破振动频率控制", "受限空间气体置换",
            "危废包装UN认证", "放射源运输容器", "土壤VOCs阈值",
            "化学锚栓拆除", "衬里剥离风险", "地下管网追踪"
        ]),

        ("危化品物流安全", [
            "UN包装气密性测试", "海运稳定剂浓度", "航空禁运清单动态",
            "多式联运温控衔接", "ADR罐车接地电阻", "铁路编组隔离距离",
            "LNG预冷速率控制", "多语言应急符号", "驾驶员瞳孔监测",
            "电子运单防篡改", "泄漏吸附材料兼容", "运输路线风险"
        ]),

        ("化工职业健康", [
            "肌肉骨骼NIOSH评估", "致癌物接触限值", "脑力负荷NASA-TLX",
            "轮班耐受基因检测", "应急救援VO2max", "心理健康PHQ-9",
            "纳米颗粒ELPI监测", "次声波防护", "防化服热应激",
            "化学性眼损伤", "呼吸器适合性", "工效学工具人因"
        ]),

        ("化工环保安全", [
            "碳足迹LCA边界", "脱硫废水膜蒸馏", "SCR氨逃逸在线",
            "LDAR光学成像", "恶臭电子鼻阵列", "环境损害量化",
            "生态红线无人机", "环评变更情景", "环境信用修复",
            "VOCs治理LEL", "废水毒性鱼类", "污泥重金属稳定"
        ]),

        ("化工AI安全", [
            "数字孪生机理嵌入", "异常检测贡献度", "视觉识别对抗",
            "数据漂移概念漂移", "人机协作数字孪生", "决策追溯SHAP",
            "联邦学习梯度保护", "边缘计算时延约束", "过拟合工艺识别",
            "强化学习工艺优化", "知识图谱推理", "AI腐蚀预测"
        ]),

        ("化工过程安全", [
            "HAZOP偏差矩阵", "LOPA场景切片", "SIL验算蒙特卡洛",
            "泄放系统两相流", "联锁旁路时间", "本质安全层独立",
            "人为失误THERP", "多米诺概率计算", "安全仪表共因",
            "保护层分析", "安全阀口径", "泄爆面设计"
        ]),

        ("化工供应链安全", [
            "供应商过程安全", "运输实时PPM", "原料杂质谱",
            "替代源HAZID", "分包商PSM", "地缘风险指数",
            "海关SDS验证", "包装UN兼容", "应急供应半径",
            "关键物料清单", "运输中断", "绿色供应商"
        ]),

        ("化工碳中和安全", [
            "CCUS管道腐蚀", "电解制氢膜渗透", "生物质粉尘MIT",
            "碳汇林虫害", "甲烷红外监测", "碳捕集溶剂",
            "绿氨工艺优化", "储能热管理", "碳交易区块链",
            "工艺碳强度", "碳封存监测", "碳会计审计"
        ])
    ],
    "TEMPLATES": [  # 扩展的提示模板（20种不同提问方式）
        "请从{aspect}角度提出一个关于{topic}的专业问题，要求：1）疑问句 2）聚焦实际操作 3）无解释",
        "假设您是现场工程师，遇到{topic}相关的{aspect}问题，您会如何提问？",
        "基于最新国标要求，关于{topic}的{aspect}方面，请生成一个技术问题",
        "针对{topic}中的{aspect}场景，提出一个包含具体情境的疑问句",
        "用'如何'型疑问句提出一个{topic}{aspect}相关的问题",
        "从事故预防角度，提出{topic}领域{aspect}方面的疑问",
        "请用'哪些'开头，提出一个关于{topic}{aspect}的多选型问题",
        "结合典型事故案例，生成一个{topic}{aspect}相关的分析型问题",
        "请设计一个包含'为什么'的{topic}{aspect}理论性问题",
        "从经济性角度，提出{topic}{aspect}相关的优化改进型问题"
    ],
    "INVALID_PATTERNS": [
        r"请用代码|示意图|定义|包括以下|例如|\d\.|选择",
        r"总结|要点|步骤|流程|注意事项|建议|可分为",
        r"[\(（][^\)）]{10,}[\)）]",  # 过滤长括号内容
        r".{50,}的[吗呢吧]"  # 过滤过长疑问句
    ]
}

# === 预编译正则表达式 ===
INVALID_REGEX = [re.compile(p) for p in API_CONFIG["INVALID_PATTERNS"]]
QUESTION_END_REGEX = re.compile(r'[?？]$')


# === 核心引擎 ===
class HighSpeedGenerator:
    def __init__(self):
        self.lock = threading.Lock()
        self.seen = set()
        self.questions = []
        self.stats = {"total": 0, "valid": 0, "retries": 0}

        # 主题缓存优化
        self.topic_pool = []
        for topic, aspects in API_CONFIG["TOPICS"]:
            self.topic_pool.extend([(topic, a) for a in aspects])

        # 模板缓存
        self.templates = API_CONFIG["TEMPLATES"]

    def generate_prompt(self):
        """优化提示生成速度"""
        topic, aspect = random.choice(self.topic_pool)
        return random.choice(self.templates).format(topic=topic, aspect=aspect)

    def validate_question(self, question):
        """极速验证管道"""
        # 快速失败机制
        if not (25 <= len(question) <= 75):
            return False
        if not QUESTION_END_REGEX.search(question):
            return False
        if any(rgx.search(question) for rgx in INVALID_REGEX):
            return False
        return True

    def api_request(self):
        """高性能API请求单元"""
        for retry in range(2):  # 减少重试次数
            try:
                response = requests.post(
                    API_CONFIG["ENDPOINT"],
                    headers={"Authorization": f"Bearer {API_CONFIG['KEY']}"},
                    json={
                        "model": "glm-4-flash",
                        "messages": [
                            {"role": "system", "content": "生成化工安全疑问句"},
                            {"role": "user", "content": self.generate_prompt()}
                        ],
                        "temperature": API_CONFIG["TEMP"],
                        "max_tokens": API_CONFIG["MAX_TOKENS"]
                    },
                    timeout=15  # 缩短超时时间
                )

                # 动态速率控制
                remain = int(response.headers.get('X-RateLimit-Remaining', 100))
                if remain < 15:
                    time.sleep(max(0.5, 15 / remain))

                content = response.json()["choices"][0]["message"]["content"].strip()
                return content.replace('"', '')

            except Exception as e:
                time.sleep(0.3 * (retry + 1))
                self.stats["retries"] += 1
        return None

    def process_batch(self):
        """高速批量处理"""
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.api_request)
                       for _ in range(API_CONFIG["BATCH_SIZE"])]

            batch = []
            for future in as_completed(futures):
                if (resp := future.result()) and self.validate_question(resp):
                    q_hash = hashlib.sha1(resp.encode()).hexdigest()  # 更快的哈希
                    if q_hash not in self.seen:
                        batch.append(resp)
                        self.seen.add(q_hash)
                        self.stats["valid"] += 1
                    self.stats["total"] += 1
            return batch

    def save_results(self):
        """安全编码的保存方法"""
        with open(API_CONFIG["OUTPUT_FILE"], "w", encoding="utf-8") as f:
            f.write("questions = [\n")
            # 使用json.dumps确保正确转义
            formatted = [f'    {json.dumps(question, ensure_ascii=False)}'
                         for question in self.questions]
            f.write(",\n".join(formatted))
            f.write("\n]")


# === 优化后的主程序 ===
def main():
    gen = HighSpeedGenerator()
    target = 30000  # 建议批量生成

    with tqdm(total=target, desc="高速生成", unit="问") as pbar:
        with ThreadPoolExecutor(max_workers=API_CONFIG["CONCURRENCY"]) as executor:
            futures = []

            # 动态任务管理
            while len(gen.questions) < target:
                # 保持任务队列充足
                while len(futures) < API_CONFIG["CONCURRENCY"] * 2:
                    futures.append(executor.submit(gen.process_batch))

                # 处理完成的任务
                done, _ = as_completed(futures), []
                for f in done:
                    if (batch := f.result()):
                        with gen.lock:
                            add_num = min(target - len(gen.questions), len(batch))
                            gen.questions.extend(batch[:add_num])
                            pbar.update(add_num)
                    futures.remove(f)

    gen.save_results()
    print(f"生成统计: 总请求{gen.stats['total']} 有效率{gen.stats['valid'] / gen.stats['total']:.1%}")


if __name__ == "__main__":
    main()
