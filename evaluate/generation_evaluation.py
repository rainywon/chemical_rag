"""
生成模块评估脚本
评估指标：忠实度(Faithfulness)和答案相关性(Answer Relevancy)
使用RAGAS框架进行评估
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd
import numpy as np

# 引入RAGAS评估库
from ragas.metrics import faithfulness, answer_relevancy
from ragas.metrics.critique import harmfulness

# 引入系统配置和RAG系统
from config import Config
from rag_system import RAGSystem

# 设置日志
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class GenerationEvaluator:
    """RAG系统生成模块评估器"""
    
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
        
        # 初始化RAGAS评估指标
        self.faithfulness_metric = faithfulness
        self.relevancy_metric = answer_relevancy
        self.harmfulness_metric = harmfulness
        
    def _load_test_data(self) -> List[Dict[str, Any]]:
        """加载测试数据集"""
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def evaluate_faithfulness(self, question: str, answer: str, contexts: List[str]) -> float:
        """
        评估回答的忠实度
        
        Args:
            question: 用户问题
            answer: 生成的回答
            contexts: 检索的上下文文本列表
            
        Returns:
            忠实度得分 (0~1)
        """
        try:
            # 创建评估数据
            eval_data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts]
            }
            df = pd.DataFrame(eval_data)
            
            # 计算忠实度得分
            result = self.faithfulness_metric.score(df)
            return float(result['faithfulness'].iloc[0])
            
        except Exception as e:
            logger.error(f"评估忠实度时出错: {str(e)}")
            return 0.0
    
    def evaluate_relevancy(self, question: str, answer: str) -> float:
        """
        评估答案相关性
        
        Args:
            question: 用户问题
            answer: 生成的回答
            
        Returns:
            相关性得分 (0~1)
        """
        try:
            # 创建评估数据
            eval_data = {
                "question": [question],
                "answer": [answer],
            }
            df = pd.DataFrame(eval_data)
            
            # 计算相关性得分
            result = self.relevancy_metric.score(df)
            return float(result['answer_relevancy'].iloc[0])
            
        except Exception as e:
            logger.error(f"评估答案相关性时出错: {str(e)}")
            return 0.0
    
    def evaluate(self) -> Dict[str, Any]:
        """
        评估生成模块性能
        
        Returns:
            评估结果字典
        """
        results = {
            "faithfulness": [],
            "relevancy": [],
            "detailed_results": []
        }
        
        logger.info(f"开始评估生成模块性能，测试集大小: {len(self.test_data)}")
        
        for item in tqdm(self.test_data):
            question = item["question"]
            reference_answer = item.get("reference_answer", "")
            
            try:
                # 使用RAG系统生成回答
                answer, retrieved_docs, _ = self.rag_system.answer_query(question)
                
                # 提取检索到的上下文
                contexts = []
                for doc in retrieved_docs:
                    if "text_chunk" in doc:
                        contexts.append(doc["text_chunk"])
                    else:
                        logger.warning(f"检索结果中缺少文本内容: {doc}")
                
                # 评估忠实度
                faith_score = self.evaluate_faithfulness(question, answer, contexts)
                
                # 评估答案相关性
                rel_score = self.evaluate_relevancy(question, answer)
                
                # 记录结果
                results["faithfulness"].append(faith_score)
                results["relevancy"].append(rel_score)
                
                # 详细记录
                detailed_result = {
                    "question": question,
                    "answer": answer,
                    "reference_answer": reference_answer,
                    "faithfulness": faith_score,
                    "relevancy": rel_score
                }
                results["detailed_results"].append(detailed_result)
                
            except Exception as e:
                logger.error(f"评估问题 '{question}' 时出错: {str(e)}")
        
        # 计算平均得分
        summary = {
            "avg_faithfulness": np.mean(results["faithfulness"]),
            "avg_relevancy": np.mean(results["relevancy"]),
            "detailed_results": results["detailed_results"]
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
        logger.info("\n===== 生成模块评估结果 =====")
        logger.info(f"平均忠实度: {results['avg_faithfulness']:.4f}")
        logger.info(f"平均答案相关性: {results['avg_relevancy']:.4f}")
        
        # 保存结果
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存摘要结果
            with open(output_path, 'w', encoding='utf-8') as f:
                # 移除detailed_results以避免文件过大
                summary_results = {k: v for k, v in results.items() 
                                  if k != "detailed_results"}
                json.dump(summary_results, f, indent=2, ensure_ascii=False)
            
            # 保存详细结果到单独文件
            detailed_path = output_path.replace('.json', '_detailed.json')
            with open(detailed_path, 'w', encoding='utf-8') as f:
                json.dump({"detailed_results": results["detailed_results"]}, 
                         f, indent=2, ensure_ascii=False)
                
            logger.info(f"评估结果已保存至: {output_path}")
            logger.info(f"详细评估结果已保存至: {detailed_path}")

def main():
    parser = argparse.ArgumentParser(description="RAG系统生成模块评估")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据集路径")
    parser.add_argument("--output", type=str, default="evaluation_results/generation_results.json", 
                        help="评估结果保存路径")
    args = parser.parse_args()
    
    evaluator = GenerationEvaluator(args.config, args.test_data)
    evaluator.run_evaluation(args.output)

if __name__ == "__main__":
    main() 