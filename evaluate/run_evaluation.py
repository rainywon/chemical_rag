"""
化工安全RAG系统评估主运行程序
整合检索模块和生成模块的评估
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# 添加父目录到路径，以便导入项目模块
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from evaluate.evaluate_retrieval import RetrievalEvaluator
from evaluate.evaluate_generation import GenerationEvaluator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def run_full_evaluation(config: Config, test_data_dir: str, output_dir: str) -> Dict[str, Any]:
    """运行完整的系统评估
    
    Args:
        config: 系统配置对象
        test_data_dir: 测试数据目录
        output_dir: 输出结果目录
        
    Returns:
        包含所有评估结果的字典
    """
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # 评估检索模块
    logger.info("开始检索模块评估...")
    retrieval_evaluator = RetrievalEvaluator(config)
    retrieval_results = retrieval_evaluator.run_evaluation(
        f"{test_data_dir}/retrieval_test_data.json"
    )
    
    # 保存检索评估结果
    with open(f"{output_dir}/retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=2)
    logger.info(f"检索评估结果已保存至: {output_dir}/retrieval_results.json")
    
    # 评估生成模块
    logger.info("开始生成模块评估...")
    generation_evaluator = GenerationEvaluator(config)
    generation_results = generation_evaluator.run_evaluation(
        f"{test_data_dir}/generation_test_data.json"
    )
    
    # 保存生成评估结果
    with open(f"{output_dir}/generation_results.json", "w", encoding="utf-8") as f:
        json.dump(generation_results, f, ensure_ascii=False, indent=2)
    logger.info(f"生成评估结果已保存至: {output_dir}/generation_results.json")
    
    # 合并所有结果
    all_results = {
        "retrieval": retrieval_results,
        "generation": generation_results,
        "timestamp": Path(f"{output_dir}/evaluation_results.json").stat().st_mtime if Path(f"{output_dir}/evaluation_results.json").exists() else None
    }
    
    # 保存完整评估结果
    with open(f"{output_dir}/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"完整评估结果已保存至: {output_dir}/evaluation_results.json")
    
    return all_results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="化工安全RAG系统评估工具")
    
    # 添加命令行参数
    parser.add_argument(
        "--test_data_dir", 
        type=str, 
        default="evaluate/test_data",
        help="测试数据目录路径"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluate/results",
        help="评估结果输出目录路径"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    
    # 运行评估
    logger.info("开始进行完整系统评估...")
    results = run_full_evaluation(config, args.test_data_dir, args.output_dir)
    
    # 打印关键指标
    print("\n" + "="*50)
    print("化工安全RAG系统评估结果")
    print("="*50)
    
    # 检索模块指标
    print("\n▶ 检索模块性能:")
    print(f"  命中率@1: {results['retrieval']['hit_rate']['hit@1']:.4f}")
    print(f"  命中率@3: {results['retrieval']['hit_rate']['hit@3']:.4f}")
    print(f"  命中率@5: {results['retrieval']['hit_rate']['hit@5']:.4f}")
    print(f"  平均倒数排名(MRR): {results['retrieval']['mrr']:.4f}")
    
    # 生成模块指标
    print("\n▶ 生成模块性能:")
    print(f"  忠实度: {results['generation']['faithfulness']['faithfulness_score']:.4f}")
    print(f"  矛盾率: {results['generation']['faithfulness']['contradiction_rate']:.4f}")
    print(f"  支持率: {results['generation']['faithfulness']['support_rate']:.4f}")
    print(f"  相关性: {results['generation']['relevance']['relevance_score']:.4f}")
    
    print("\n✅ 评估完成！详细结果保存在: " + args.output_dir)

if __name__ == "__main__":
    main() 