"""
检索模块评估脚本
评估指标：命中率(Hit Rate)和平均倒数排名(MMR)
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm

# 引入系统配置和RAG系统
from config import Config
from rag_system import RAGSystem

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    """RAG系统检索模块评估器"""
    
    def __init__(self, config_path: str, test_data_path: str):
        """
        初始化评估器
        
        Args:
            config_path: 配置文件路径
            test_data_path: 测试数据集路径
        """
        self.config = Config.from_json(config_path)
        self.test_data_path = test_data_path
        self.rag_system = RAGSystem(self.config)
        self.test_data = self._load_test_data()
        
    def _load_test_data(self) -> List[Dict[str, Any]]:
        """加载测试数据集"""
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_hit_rate(self, retrieved_docs: List[Dict], 
                          relevant_docs: List[str], k: int = None) -> float:
        """
        计算命中率 (Hit Rate @ K)
        
        Args:
            retrieved_docs: 检索到的文档列表
            relevant_docs: 相关文档的ID列表
            k: 考虑的检索结果数量
            
        Returns:
            命中率 (0~1)
        """
        if not relevant_docs:
            return 0.0
            
        if k is not None:
            retrieved_docs = retrieved_docs[:k]
            
        retrieved_ids = [Path(doc.get("full_path", "")).name for doc in retrieved_docs]
        hits = sum(1 for doc_id in retrieved_ids if doc_id in relevant_docs)
        
        return hits / len(relevant_docs)
    
    def calculate_mrr(self, retrieved_docs: List[Dict], 
                     relevant_docs: List[str]) -> float:
        """
        计算平均倒数排名 (Mean Reciprocal Rank)
        
        Args:
            retrieved_docs: 检索到的文档列表
            relevant_docs: 相关文档的ID列表
            
        Returns:
            MRR值 (0~1)
        """
        if not relevant_docs or not retrieved_docs:
            return 0.0
            
        retrieved_ids = [Path(doc.get("full_path", "")).name for doc in retrieved_docs]
        
        # 找出第一个相关文档的排名
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_docs:
                return 1.0 / (i + 1)
                
        return 0.0
    
    def evaluate(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        评估检索模块性能
        
        Args:
            k_values: 要评估的k值列表
            
        Returns:
            评估结果字典
        """
        results = {
            "hit_rate": {f"@{k}": [] for k in k_values},
            "mrr": []
        }
        
        logger.info(f"开始评估检索模块性能，测试集大小: {len(self.test_data)}")
        
        for item in tqdm(self.test_data):
            question = item["question"]
            relevant_docs = item["relevant_docs"]
            
            # 调用系统检索文档
            try:
                docs, score_info = self.rag_system._retrieve_documents(question)
                references = self.rag_system._format_references(docs, score_info)
                
                # 计算各个k值下的命中率
                for k in k_values:
                    hr_at_k = self.calculate_hit_rate(references, relevant_docs, k)
                    results["hit_rate"][f"@{k}"].append(hr_at_k)
                
                # 计算MRR
                mrr = self.calculate_mrr(references, relevant_docs)
                results["mrr"].append(mrr)
                
            except Exception as e:
                logger.error(f"评估问题 '{question}' 时出错: {str(e)}")
        
        # 计算平均值
        summary = {
            "hit_rate": {k: np.mean(v) for k, v in results["hit_rate"].items()},
            "mrr": np.mean(results["mrr"])
        }
        
        return summary
        
    def run_evaluation(self, output_path: str = None):
        """
        运行评估并保存结果
        
        Args:
            output_path: 结果保存路径
        """
        results = self.evaluate()
        
        # 打印结果
        logger.info("\n===== 检索模块评估结果 =====")
        for k, hr in results["hit_rate"].items():
            logger.info(f"命中率 {k}: {hr:.4f}")
        logger.info(f"平均倒数排名 (MRR): {results['mrr']:.4f}")
        
        # 保存结果
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"评估结果已保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="RAG系统检索模块评估")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据集路径")
    parser.add_argument("--output", type=str, default="evaluation_results/retrieval_results.json", 
                        help="评估结果保存路径")
    args = parser.parse_args()
    
    evaluator = RetrievalEvaluator(args.config, args.test_data)
    evaluator.run_evaluation(args.output)

if __name__ == "__main__":
    main() 