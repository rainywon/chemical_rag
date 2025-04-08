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
    "TEMP": 0.7,  # 提高温度增加多样性
    "BATCH_SIZE": 30,
    "OUTPUT_FILE": "random_12000_questions.py",  # 修正文件名与produce_lora_data.py中引用一致
    "CONCURRENCY": 12,  # 根据CPU核心数优化
    "TOPICS": [
        ("化工现场实操", [
            "装置开停车操作", "检修安全隔离", "临时作业许可", 
            "设备密封点检查", "管线吹扫置换", "罐区安全切水",
            "阀门维修调试", "仪表零点校准", "取样标准动作", 
            "现场应急演练", "消防设施测试", "警示标识张贴",
            "工具归类存放", "安全带悬挂点", "有毒气体检测",
            "动设备润滑管理", "静设备防腐措施", "电气设备巡检",
            "工艺参数调整", "DCS操作规范", "现场交接班管理",
            "安全设施维护", "应急物资检查", "受限空间管理",
            "高处作业监护", "动火作业管控", "临时用电管理",
            "起重吊装指挥", "脚手架搭拆", "射线探伤防护"
        ]),

        ("基层员工安全", [
            "岗位安全培训", "风险辨识能力", "个人防护用品", 
            "劳动保护措施", "身心健康防护", "安全奖惩制度",
            "班组安全文化", "事故案例学习", "安全生产责任", 
            "安全隐患举报", "危险化学品知识", "班前班后会",
            "职业病防护", "安全作业证", "工作票管理",
            "新员工三级教育", "转岗培训要求", "特种作业取证",
            "安全技能竞赛", "应急能力评估", "安全行为观察",
            "安全经验分享", "安全合理化建议", "安全绩效考评",
            "安全知识竞赛", "安全技能比武", "安全文化建设",
            "安全责任落实", "安全奖惩兑现", "安全考核评价"
        ]),

        ("问题设备处理", [
            "泵轴承过热", "阀门泄漏处理", "换热器堵塞", 
            "压缩机振动", "反应釜结焦", "储罐底部沉积",
            "管道冻裂预防", "搅拌器机封", "塔板损坏", 
            "过滤器压差大", "脱硫塔腐蚀", "安全阀粘连",
            "仪表失灵判断", "控制阀卡死", "电机过载保护",
            "离心泵气蚀", "往复泵脉动", "螺杆泵磨损",
            "换热器结垢", "冷凝器泄漏", "蒸发器结焦",
            "反应器飞温", "搅拌器断轴", "塔器偏流",
            "管道振动", "法兰泄漏", "垫片失效",
            "安全阀起跳", "爆破片破裂", "呼吸阀堵塞"
        ]),

        ("化工过程安全", [
            "HAZOP偏差矩阵", "LOPA场景切片", "SIL验算蒙特卡洛",
            "泄放系统两相流", "联锁旁路时间", "本质安全层独立",
            "人为失误THERP", "多米诺概率计算", "安全仪表共因",
            "保护层分析", "安全阀口径", "泄爆面设计",
            "工艺危害分析", "变更管理评估", "开车前安全检查",
            "工艺安全信息", "机械完整性", "操作程序管理",
            "承包商管理", "事故调查分析", "应急响应计划",
            "工艺安全审计", "安全绩效指标", "安全文化建设",
            "工艺安全培训", "安全经验分享", "安全奖惩制度",
            "工艺安全标准", "安全操作规程", "安全管理制度"
        ]),

        ("危化品管理", [
            "MSDS合规性", "禁忌物料识别", "相容性管理",
            "微量分装安全", "过期化学品处理", "剧毒物管理",
            "纳米材料控制", "自反应物质存储", "临界量监测",
            "管输化学品控制", "电化学腐蚀防护", "聚合抑制剂监测",
            "危险化学品分类", "安全标签管理", "安全技术说明书",
            "储存条件控制", "使用量控制", "废弃处理规范",
            "泄漏应急处置", "火灾扑救方法", "中毒急救措施",
            "个人防护装备", "通风系统管理", "监测报警系统",
            "安全操作规程", "应急预案制定", "安全培训教育",
            "安全检查制度", "安全责任落实", "安全奖惩制度"
        ]),

        ("化工应急管理", [
            "应急预案编制", "应急演练组织", "应急物资管理",
            "事故初期处置", "人员疏散指挥", "泄漏应急处置",
            "火灾扑救方法", "中毒急救措施", "环境污染控制",
            "应急通讯保障", "应急照明系统", "应急电源管理",
            "应急指挥体系", "应急响应程序", "应急资源调配",
            "应急演练评估", "应急培训教育", "应急能力建设",
            "应急物资储备", "应急装备维护", "应急设施管理",
            "应急通讯系统", "应急照明系统", "应急电源系统",
            "应急疏散通道", "应急避难场所", "应急医疗救护",
            "应急环境监测", "应急污染控制", "应急恢复重建"
        ]),
        
        ("典型化工工艺", [
            "加氢反应控制", "聚合反应安全", "氧化工艺管理", 
            "精馏操作规程", "蒸馏系统维护", "结晶工艺参数",
            "干燥设备除尘", "萃取工艺优化", "气固反应控制", 
            "中和反应pH值", "催化剂再生", "硝化工艺冷却",
            "氯化反应抑制", "吸收塔效率", "MVR蒸发器",
            "磺化工艺控制", "胺化反应安全", "酯化工艺优化",
            "水解反应控制", "缩合反应安全", "重氮化工艺管理",
            "烷基化工艺", "酰化反应控制", "环化反应安全",
            "异构化工艺", "脱水反应控制", "脱氢反应安全",
            "羰基化工艺", "氢化反应控制", "氧化还原反应"
        ]),
        
        ("特殊作业安全", [
            "有限空间作业", "高处作业防护", "动火作业管控", 
            "吊装作业规范", "挖掘作业标准", "带电作业措施",
            "盲板抽堵作业", "射线探伤防护", "高压水清洗", 
            "低温作业防冻", "保温拆除作业", "脚手架搭建",
            "检维修作业票", "临时用电管理", "交叉作业协调",
            "受限空间监护", "高处作业监护", "动火作业监护",
            "吊装作业指挥", "挖掘作业监护", "带电作业监护",
            "盲板抽堵监护", "射线探伤监护", "高压水清洗监护",
            "低温作业监护", "保温拆除监护", "脚手架搭拆监护",
            "检维修作业监护", "临时用电监护", "交叉作业监护"
        ]),
        
        ("设备完整性", [
            "压力容器检验", "管道完整性管理", "法兰密封选型", 
            "阀门泄漏等级", "焊缝无损检测", "设备寿命评估",
            "腐蚀监测点位", "在线检测技术", "管线防腐措施", 
            "热交换器检查", "储罐底板检测", "垫片材质选择",
            "机泵振动分析", "压力表校验", "安全附件测试",
            "设备台账管理", "设备档案管理", "设备状态监测",
            "设备维护保养", "设备检修管理", "设备更新改造",
            "设备故障分析", "设备寿命预测", "设备风险评估",
            "设备安全评估", "设备可靠性分析", "设备可用性分析",
            "设备维护策略", "设备检修策略", "设备更新策略"
        ])
    ],
    "TEMPLATES": [  # 扩展的提示模板（更贴近用户实际提问）
        # 急切寻求解决方法的问题
        "我们工厂在{topic}过程中遇到了{aspect}问题，该怎么解决？",
        "最近在处理{topic}时，{aspect}出现异常，有什么应对方法吗？",
        "我是一名化工操作工，昨天在{topic}过程中发现{aspect}异常，这是什么原因？",
        
        # 寻求最佳实践
        "有没有关于{topic}中{aspect}的最佳实践或推荐标准？",
        "我们想改进{topic}的{aspect}，业内有什么成功案例可以参考？",
        "作为化工安全主管，怎样提高{topic}过程中的{aspect}管理水平？",
        
        # 对比型问题
        "{topic}中采用A方法和B方法处理{aspect}问题，哪个更安全？",
        "传统{topic}和新型{topic}在{aspect}方面有什么区别？",
        
        # 验证自己理解的问题
        "我理解{topic}需要控制{aspect}在一定范围内，这样做对吗？",
        "听说{topic}过程中{aspect}可能导致安全事故，真的是这样吗？",
        
        # 处理紧急情况
        "{topic}设备突然出现{aspect}报警，需要紧急停车吗？",
        "发现{topic}区域有{aspect}泄漏，应该采取哪些应急措施？",
        
        # 日常操作疑问
        "{topic}日常维护中，{aspect}检查项有哪些容易被忽略？",
        "新入职员工在{topic}岗位上，应该特别注意哪些{aspect}风险？",
        
        # 法规合规问题
        "最新安全法规对{topic}的{aspect}有什么新要求？",
        "{topic}的{aspect}指标需要达到什么标准才能通过审核？",
        
        # 设备选型问题
        "我们需要更换{topic}设备，在{aspect}方面应该考虑哪些因素？",
        "选购{topic}系统时，如何评估其{aspect}性能？",
        
        # 事故分析问题
        "前段时间某厂{topic}{aspect}事故的根本原因是什么？",
        "{topic}发生{aspect}故障的常见原因有哪些？",
        
        # 培训相关问题
        "如何对新员工进行{topic}的{aspect}培训？",
        "我需要为班组准备{topic}{aspect}的安全培训，有什么重点内容？",
        
        # 工艺优化问题
        "{topic}工艺中{aspect}参数如何优化才能兼顾安全和效率？",
        "我们{topic}工序的{aspect}指标波动较大，如何稳定控制？",
        
        # 经验传承问题
        "老师傅们在{topic}中处理{aspect}有什么经验技巧？",
        "作为新手，如何快速掌握{topic}中{aspect}的关键控制点？",
        
        # 异常判断问题
        "{topic}过程中{aspect}出现这种情况算正常吗？",
        "在{topic}中，{aspect}数值达到多少才需要报警？",
        
        # 风险评估问题
        "我们要进行{topic}的{aspect}变更，需要评估哪些风险？",
        "{topic}的{aspect}风险等级应该如何确定？",
        
        # 现场实操问题
        "{topic}现场操作中，{aspect}的具体步骤是怎样的？",
        "进行{topic}时，{aspect}操作的安全要点有哪些？",
        
        # 设备故障问题
        "{topic}的{aspect}设备总是出故障，可能是什么原因？",
        "我们{topic}区域的{aspect}系统频繁报警，怎么排查？",
        
        # 人员安全问题
        "{topic}过程中如何确保员工在{aspect}方面的人身安全？",
        "发生{topic}{aspect}事故后，现场人员应如何自保？",
        
        # 工程改造问题
        "我们准备对{topic}进行改造，{aspect}方面需要注意什么？",
        "{topic}装置的{aspect}系统升级有什么建议？",
        
        # 材料选择问题
        "{topic}中接触{aspect}的材料应该选用什么材质？",
        "用于{topic}的{aspect}设备，材质选型标准是什么？",
        
        # 运行周期问题
        "{topic}工艺中{aspect}多久检查一次为宜？",
        "我们{topic}的{aspect}维保周期该如何确定？",
        
        # 环保问题
        "{topic}过程中{aspect}产生的废弃物如何安全处理？",
        "{topic}工艺的{aspect}环节如何降低环境影响？",
        
        # 岗位职责问题
        "在{topic}岗位上，处理{aspect}问题是谁的责任？",
        "{topic}中的{aspect}异常应该由哪个部门处理？"
    ],
    "INVALID_PATTERNS": [
        r"请用代码|示意图|定义|包括以下|例如|\d\.|选择",
        r"总结|要点|步骤|流程|注意事项|建议|可分为",
        r"[\(（][^\)）]{10,}[\)）]",  # 过滤长括号内容
        r".{100,}的[吗呢吧]",  # 修改为允许更长的问题
        r"^请问|^请告诉我|^我想知道",  # 过滤太过正式的开头
        r"感谢|谢谢"  # 过滤结尾客套语
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
        """验证问题的适用性和真实性"""
        # 长度检查 - 允许更长的问题
        if not (20 <= len(question) <= 160):  # 扩大允许范围
            return False
            
        # 问号检查 - 放宽要求，允许没有问号结尾的陈述句问题
        # 很多真实用户提问可能是"遇到了xxx问题，求解决方案"这样的形式
        contains_question_mark = QUESTION_END_REGEX.search(question)
        contains_question_words = re.search(r'怎么|如何|什么|为什么|哪些|多少|是否|可以|能否|需要|该不该|建议|方法|步骤|标准|操作', question)
        
        if not (contains_question_mark or contains_question_words):
            return False
            
        # 其他验证规则
        if any(rgx.search(question) for rgx in INVALID_REGEX):
            return False
            
        # 真实性检查 - 确保问题听起来像真人提问
        if re.search(r'我要求你|请生成|请提供一个|我命令你', question):
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
                            {"role": "system", "content": "你是一名化工安全领域的工程师。生成一个真实的化工安全问题，使其听起来像是一线工作人员在实际工作中提出的。问题应该简洁、直接，体现出工作中的紧迫性或实际问题，不要使用过于正式或教科书式的语言。可以包含工厂常见的简写术语。"},
                            {"role": "user", "content": self.generate_prompt()}
                        ],
                        "temperature": API_CONFIG["TEMP"],  # 使用配置温度
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
    target = 12000  # 建议批量生成

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
