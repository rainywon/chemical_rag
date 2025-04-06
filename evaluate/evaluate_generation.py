"""
评估RAG系统的生成模块性能
实现了忠实度(Faithfulness)和答案相关性(Answer Relevance)指标
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
from collections import Counter

# 添加父目录到路径，以便导入项目模块
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from rag_system import RAGSystem

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class GenerationEvaluator:
    """生成模块评估器"""
    
    def __init__(self, config: Config):
        """初始化评估器
        
        Args:
            config: 系统配置对象
        """
        self.config = config
        self.rag_system = RAGSystem(config)
        logger.info("生成评估器初始化完成")
    
    def _extract_key_facts(self, text: str) -> List[str]:
        """从文本中提取关键事实
        
        Args:
            text: 输入文本
            
        Returns:
            关键事实列表
        """
        # 基于句子分割
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 过滤太短的句子
        facts = [s for s in sentences if len(s) > 10]
        
        return facts
    
    def _contains_contradiction(self, fact: str, reference_text: str) -> bool:
        """检测事实是否与参考文本存在矛盾
        
        当前实现使用了简单的否定检测方法，实际应用中应考虑更复杂的方法
        
        Args:
            fact: 待检查的事实
            reference_text: 参考文本
            
        Returns:
            是否存在矛盾
        """
        # 简化的矛盾检测：检查事实中的关键信息在参考文本中是否有对立表述
        # 例如："A不是B" vs "A是B"
        
        # 提取否定表达
        negation_patterns = [
            r'不是', r'不能', r'不可以', r'没有', r'不会', r'不应该',
            r'不得', r'禁止', r'严禁', r'不允许', r'不宜', r'不适合'
        ]
        
        # 检查否定表达
        for pattern in negation_patterns:
            if pattern in fact:
                # 提取被否定的内容
                parts = fact.split(pattern)
                if len(parts) > 1:
                    affirmative = parts[1].strip()
                    # 在参考文本中搜索肯定表述
                    if len(affirmative) > 2 and affirmative in reference_text:
                        # 检查参考文本中是否有肯定表述
                        if pattern not in reference_text[
                            max(0, reference_text.find(affirmative) - 10):
                            reference_text.find(affirmative)
                        ]:
                            return True
        
        return False
    
    def evaluate_faithfulness(self, test_data: List[Dict]) -> Dict[str, float]:
        """评估忠实度 - 生成的答案是否符合检索文档中的事实
        
        Args:
            test_data: 包含问题的测试数据
            
        Returns:
            忠实度评估结果
        """
        logger.info("开始评估忠实度...")
        
        contradiction_count = 0
        support_count = 0
        total_facts = 0
        
        for item in test_data:
            query = item["question"]
            
            # 使用RAG系统生成答案
            answer, references, metadata = self.rag_system.answer_query(query)
            
            # 跳过失败的查询
            if metadata["status"] != "success":
                logger.warning(f"查询失败，跳过: {query}")
                continue
            
            # 提取答案中的关键事实
            facts = self._extract_key_facts(answer)
            total_facts += len(facts)
            
            # 构建参考文本
            reference_text = " ".join([ref["content"] for ref in references])
            
            # 检查每个事实
            for fact in facts:
                # 检查矛盾
                if self._contains_contradiction(fact, reference_text):
                    contradiction_count += 1
                # 检查支持 - 简单估计支持度（事实中的关键词是否在参考文本中出现）
                elif self._check_fact_support(fact, reference_text):
                    support_count += 1
        
        # 计算指标
        contradiction_rate = contradiction_count / total_facts if total_facts > 0 else 0
        support_rate = support_count / total_facts if total_facts > 0 else 0
        
        results = {
            "contradiction_rate": contradiction_rate,
            "support_rate": support_rate,
            "faithfulness_score": 1 - contradiction_rate + support_rate / 2,  # 自定义综合指标
            "total_facts": total_facts
        }
        
        logger.info(f"忠实度评估结果: {results}")
        return results
    
    def _check_fact_support(self, fact: str, reference_text: str) -> bool:
        """检查事实是否被参考文本支持
        
        Args:
            fact: 待检查的事实
            reference_text: 参考文本
            
        Returns:
            是否被支持
        """
        # 分词并获取关键词（简化版）
        keywords = [w for w in fact.split() if len(w) > 1]
        
        # 计算多少关键词在参考文本中出现
        matches = sum(1 for word in keywords if word in reference_text)
        
        # 如果超过60%的关键词在参考文本中，认为事实得到支持
        if len(keywords) > 0 and matches / len(keywords) >= 0.6:
            return True
        return False
    
    def _compute_word_overlap(self, text1: str, text2: str) -> float:
        """计算词重叠率
        
        Args:
            text1, text2: 待比较的文本
            
        Returns:
            词重叠率
        """
        words1 = Counter(text1.split())
        words2 = Counter(text2.split())
        
        # 计算交集
        intersection = sum((words1 & words2).values())
        union = sum((words1 | words2).values())
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def evaluate_answer_relevance(self, test_data: List[Dict]) -> Dict[str, float]:
        """评估答案相关性 - 生成的答案与问题的相关程度
        
        Args:
            test_data: 包含问题和标准答案的测试数据
            
        Returns:
            相关性评估结果
        """
        logger.info("开始评估答案相关性...")
        
        question_overlap_scores = []
        gold_overlap_scores = []  # 与黄金标准答案的重叠
        context_relevance_scores = []
        
        for item in test_data:
            query = item["question"]
            gold_answer = item.get("gold_answer", "")  # 黄金标准答案
            
            # 使用RAG系统生成答案
            answer, references, metadata = self.rag_system.answer_query(query)
            
            # 跳过失败的查询
            if metadata["status"] != "success":
                logger.warning(f"查询失败，跳过: {query}")
                continue
            
            # 1. 问题-答案词重叠（共同主题词）
            question_overlap = self._compute_word_overlap(query, answer)
            question_overlap_scores.append(question_overlap)
            
            # 2. 与黄金标准答案比较（如果存在）
            if gold_answer:
                gold_overlap = self._compute_word_overlap(answer, gold_answer)
                gold_overlap_scores.append(gold_overlap)
            
            # 3. 上下文相关性（答案包含的检索内容）
            context_text = " ".join([ref["content"] for ref in references])
            context_overlap = self._compute_word_overlap(answer, context_text)
            context_relevance_scores.append(context_overlap)
        
        # 计算平均分数
        avg_question_overlap = np.mean(question_overlap_scores)
        avg_context_relevance = np.mean(context_relevance_scores)
        
        results = {
            "question_overlap": avg_question_overlap,
            "context_relevance": avg_context_relevance,
        }
        
        # 如果有黄金标准答案
        if gold_overlap_scores:
            avg_gold_overlap = np.mean(gold_overlap_scores)
            results["gold_answer_overlap"] = avg_gold_overlap
        
        # 计算综合分数
        results["relevance_score"] = (
            0.3 * avg_question_overlap + 
            0.7 * avg_context_relevance
        )
        
        if "gold_answer_overlap" in results:
            # 重新计算，加入黄金标准
            results["relevance_score"] = (
                0.2 * avg_question_overlap + 
                0.5 * avg_context_relevance + 
                0.3 * results["gold_answer_overlap"]
            )
        
        logger.info(f"答案相关性评估结果: {results}")
        return results
    
    def run_evaluation(self, test_data_path: str) -> Dict[str, Any]:
        """运行完整评估
        
        Args:
            test_data_path: 测试数据文件路径
            
        Returns:
            评估结果
        """
        # 加载测试数据
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        logger.info(f"加载了{len(test_data)}条测试数据")
        
        # 运行评估
        faithfulness = self.evaluate_faithfulness(test_data)
        relevance = self.evaluate_answer_relevance(test_data)
        
        # 合并结果
        results = {
            "faithfulness": faithfulness,
            "relevance": relevance
        }
        
        return results

if __name__ == "__main__":
    # 加载配置
    config = Config()
    
    # 创建评估器
    evaluator = GenerationEvaluator(config)
    
    # 运行评估
    results = evaluator.run_evaluation("evaluate/test_data/generation_test_data.json")
    
    # 保存结果
    result_path = Path("evaluate/results/generation_results.json")
    result_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估结果已保存至: {result_path}") 