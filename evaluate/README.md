# 化工安全RAG系统评估模块

本目录包含用于评估化工安全RAG系统性能的工具和测试数据。

## 目录结构

```
evaluate/
├── README.md                   # 本说明文档
├── run_evaluation.py           # 主评估运行程序
├── evaluate_retrieval.py       # 检索模块评估工具
├── evaluate_generation.py      # 生成模块评估工具
├── test_data/                  # 测试数据目录
│   ├── retrieval_test_data.json   # 检索模块测试数据
│   └── generation_test_data.json  # 生成模块测试数据
└── results/                    # 评估结果保存目录（自动创建）
```

## 评估指标

### 检索模块评估

1. **命中率 (Hit Rate)**
   - Hit@1: 首位检索结果中包含相关文档的比例
   - Hit@3: 前三位检索结果中包含相关文档的比例
   - Hit@5: 前五位检索结果中包含相关文档的比例
   - Hit@10: 前十位检索结果中包含相关文档的比例

2. **平均倒数排名 (Mean Reciprocal Rank, MRR)**
   - 对于每个查询，计算第一个相关文档排名的倒数，然后取平均值
   - 值越接近1表示相关文档排名越靠前

### 生成模块评估

1. **忠实度 (Faithfulness)**
   - 评估生成内容是否与检索到的文档内容一致
   - 检测生成内容中是否存在与检索文档矛盾的信息
   - 计算支持率：生成内容在检索文档中能找到支持证据的比例

2. **答案相关性 (Answer Relevance)**
   - 评估生成内容与问题的相关度
   - 计算生成内容与标准答案的重叠程度
   - 计算生成内容与上下文检索文档的匹配程度

## 使用方法

### 运行完整评估

```bash
python evaluate/run_evaluation.py
```

### 参数说明

- `--test_data_dir`: 测试数据目录路径，默认为 "evaluate/test_data"
- `--output_dir`: 结果输出目录路径，默认为 "evaluate/results"

例如：
```bash
python evaluate/run_evaluation.py --test_data_dir custom_data --output_dir custom_results
```

## 测试数据格式

### 检索测试数据格式

```json
[
  {
    "question": "问题文本",
    "relevant_docs": ["文档路径1", "文档路径2"]
  },
  ...
]
```

### 生成测试数据格式

```json
[
  {
    "question": "问题文本",
    "gold_answer": "标准答案文本（可选）"
  },
  ...
]
```

## 评估结果示例

```json
{
  "retrieval": {
    "hit_rate": {
      "hit@1": 0.75,
      "hit@3": 0.85,
      "hit@5": 0.95,
      "hit@10": 1.0
    },
    "mrr": 0.82
  },
  "generation": {
    "faithfulness": {
      "contradiction_rate": 0.05,
      "support_rate": 0.85,
      "faithfulness_score": 0.95,
      "total_facts": 120
    },
    "relevance": {
      "question_overlap": 0.32,
      "context_relevance": 0.75,
      "gold_answer_overlap": 0.68,
      "relevance_score": 0.78
    }
  }
}
``` 