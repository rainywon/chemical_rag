"""
评估RAG系统的检索模块性能
实现了命中率(Hit Rate)和平均倒数排名(Mean Reciprocal Rank, MRR)指标
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

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

class RetrievalEvaluator:
    """检索模块评估器"""
    
    def __init__(self, config: Config):
        """初始化评估器
        
        Args:
            config: 系统配置对象
        """
        self.config = config
        self.rag_system = RAGSystem(config)
        logger.info("检索评估器初始化完成")
    
    def evaluate_hit_rate(self, test_data: List[Dict]) -> Dict[str, float]:
        """评估命中率
        
        Args:
            test_data: 包含问题和相关文档的测试数据
            
        Returns:
            不同K值下的命中率结果
        """
        logger.info("开始评估命中率...")
        k_values = [1, 3, 5, 10]  # 不同K值的评估点
        hits = {f"hit@{k}": 0 for k in k_values}
        total = len(test_data)
        
        for item in test_data:
            query = item["question"]
            relevant_docs = set(item["relevant_docs"])  # 真实相关文档
            
            # 使用RAG系统进行检索
            retrieved_docs, _ = self.rag_system._retrieve_documents(query)
            
            # 获取检索文档路径
            retrieved_paths = [doc.metadata.get("source", "") for doc in retrieved_docs]
            
            # 计算不同K值下的命中情况
            for k in k_values:
                top_k_docs = set(retrieved_paths[:k])
                if any(doc in relevant_docs for doc in top_k_docs):
                    hits[f"hit@{k}"] += 1
        
        # 计算命中率
        results = {metric: count / total for metric, count in hits.items()}
        
        logger.info(f"命中率评估结果: {results}")
        return results
    
    def evaluate_mrr(self, test_data: List[Dict]) -> float:
        """评估平均倒数排名(MRR)
        
        Args:
            test_data: 包含问题和相关文档的测试数据
            
        Returns:
            MRR得分
        """
        logger.info("开始评估MRR...")
        reciprocal_ranks = []
        
        for item in test_data:
            query = item["question"]
            relevant_docs = set(item["relevant_docs"])  # 真实相关文档
            
            # 使用RAG系统进行检索
            retrieved_docs, _ = self.rag_system._retrieve_documents(query)
            
            # 获取检索文档路径
            retrieved_paths = [doc.metadata.get("source", "") for doc in retrieved_docs]
            
            # 计算倒数排名
            rank = 0
            for i, doc_path in enumerate(retrieved_paths):
                if doc_path in relevant_docs:
                    # 找到第一个相关文档的位置(从1开始计数)
                    rank = 1 / (i + 1)  
                    break
            
            reciprocal_ranks.append(rank)
        
        # 计算MRR
        mrr = np.mean(reciprocal_ranks)
        logger.info(f"MRR评估结果: {mrr:.4f}")
        return mrr
    
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
        hit_rates = self.evaluate_hit_rate(test_data)
        mrr = self.evaluate_mrr(test_data)
        
        # 合并结果
        results = {
            "hit_rate": hit_rates,
            "mrr": mrr
        }
        
        return results

if __name__ == "__main__":
    # 加载配置
    config = Config()
    
    # 创建评估器
    evaluator = RetrievalEvaluator(config)
    
    # 运行评估
    results = evaluator.run_evaluation("evaluate/test_data/retrieval_test_data.json")
    
    # 保存结果
    result_path = Path("evaluate/results/retrieval_results.json")
    result_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估结果已保存至: {result_path}") 