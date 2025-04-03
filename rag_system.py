# 导入必要的库和模块
import json
import logging  # 日志记录模块
from pathlib import Path  # 路径处理库
from typing import Generator, Optional, List, Tuple, Dict, Any  # 类型提示支持
import warnings  # 警告处理
import torch  # PyTorch深度学习框架
from langchain_community.vectorstores import FAISS  # FAISS向量数据库集成
from langchain_core.documents import Document  # 文档对象定义
from langchain_core.embeddings import Embeddings  # 嵌入模型接口
from langchain_ollama import OllamaLLM  # Ollama语言模型集成
from rank_bm25 import BM25Okapi  # BM25检索算法
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # Transformer模型
from config import Config  # 自定义配置文件
from build_vector_store import VectorDBBuilder  # 向量数据库构建器
import numpy as np  # 数值计算库
import jieba  # 中文分词库

# 禁用不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)

# 配置日志记录器
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class RAGSystem:
    """RAG问答系统，支持文档检索和生成式问答

    特性：
    - 自动管理向量数据库生命周期
    - 支持流式生成和同步生成
    - 可配置的检索策略
    - 完善的错误处理
    """

    def __init__(self, config: Config):
        """初始化RAG系统

        :param config: 包含所有配置参数的Config对象
        """
        self.config = config  # 保存配置对象
        self.vector_store: Optional[FAISS] = None  # FAISS向量数据库实例
        self.llm: Optional[OllamaLLM] = None  # Ollama语言模型实例
        self.embeddings: Optional[Embeddings] = None  # 嵌入模型实例
        self.rerank_model = None  # 重排序模型
        self.vector_db_build = VectorDBBuilder(config)  # 向量数据库构建器实例

        # 初始化各个组件
        self._init_logging()  # 初始化日志配置
        self._init_embeddings()  # 初始化嵌入模型
        self._init_vector_store()  # 初始化向量数据库
        self._init_bm25_retriever()  # 初始化BM25检索器
        self._init_llm()  # 初始化大语言模型
        self._init_rerank_model()  # 初始化重排序模型

    def _tokenize(self, text: str) -> List[str]:
        """专业中文分词处理
        :param text: 待分词的文本
        :return: 分词后的词项列表
        """
        return [word for word in jieba.cut(text) if word.strip()]

    def _init_logging(self):
        """初始化日志配置"""
        logging.basicConfig(
            level=logging.INFO,  # 日志级别设为INFO
            format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
            handlers=[logging.StreamHandler()]  # 输出到控制台
        )

    def _init_embeddings(self):
        """初始化嵌入模型"""
        try:
            logger.info("🔧 正在初始化嵌入模型...")
            # 通过构建器创建嵌入模型实例
            self.embeddings = self.vector_db_build.create_embeddings()
            logger.info("✅ 嵌入模型初始化完成")
        except Exception as e:
            logger.error("❌ 嵌入模型初始化失败")
            raise RuntimeError(f"无法初始化嵌入模型: {str(e)}")

    def _init_vector_store(self):
        """初始化向量数据库"""
        try:
            vector_path = Path(self.config.vector_db_path)  # 获取向量库路径

            # 检查现有向量数据库是否存在
            if vector_path.exists():
                logger.info("🔍 正在加载现有向量数据库...")
                if not self.embeddings:
                    raise ValueError("嵌入模型未初始化")

                # 加载本地FAISS数据库
                self.vector_store = FAISS.load_local(
                    folder_path=str(vector_path),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True  # 允许加载旧版本序列化数据
                )
                logger.info(f"✅ 已加载向量数据库：{vector_path}")
            else:
                # 构建新向量数据库
                logger.warning("⚠️ 未找到现有向量数据库，正在构建新数据库...")
                self.vector_store = self.vector_db_build.build_vector_store()
                logger.info(f"✅ 新建向量数据库已保存至：{vector_path}")
        except Exception as e:
            logger.error("❌ 向量数据库初始化失败")
            raise RuntimeError(f"无法初始化向量数据库: {str(e)}")

    def _init_rerank_model(self):
        """初始化重排序模型"""
        try:
            logger.info("🔧 正在初始化rerank模型...")
            # 从HuggingFace加载预训练模型和分词器
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.rerank_model_path
            )
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(self.config.rerank_model_path)
            logger.info("✅ rerank模型初始化完成")
        except Exception as e:
            logger.error(f"❌ rerank模型初始化失败: {str(e)}")
            raise RuntimeError(f"无法初始化rerank模型: {str(e)}")

    def _init_llm(self):
        """初始化Ollama大语言模型"""
        try:
            logger.info("🚀 正在初始化Ollama模型...")
            # 创建OllamaLLM实例
            self.llm = OllamaLLM(
                model="deepseek-r1:1.5b",  # 模型名称
                base_url=self.config.ollama_base_url,  # Ollama服务地址
                temperature=self.config.llm_temperature,  # 温度参数控制随机性
                num_predict=self.config.llm_max_tokens,  # 最大生成token数
                system="你是一位化工安全专家，请专业且准确地回答问题。",  # 系统提示词
                stop=["<|im_end|>"]  # 停止生成标记
            )

            # 测试模型连接
            test_prompt = "测试连接"
            self.llm.invoke(test_prompt)
            logger.info("✅ Ollama模型初始化完成")
        except Exception as e:
            logger.error(f"❌ Ollama模型初始化失败: {str(e)}")
            raise RuntimeError(f"无法初始化Ollama模型: {str(e)}")

    def _init_bm25_retriever(self):
        """初始化BM25检索器（改进版）"""
        try:
            logger.info("🔧 正在初始化BM25检索器...")

            # 验证向量库是否包含文档
            if not self.vector_store.docstore._dict:
                raise ValueError("向量库中无可用文档")

            # 从向量库加载所有文档内容
            all_docs = self.vector_store.docstore._dict.values()
            self.bm25_docs = [doc.page_content for doc in all_docs]
            self.doc_metadata = [doc.metadata for doc in all_docs]

            # 中文分词处理
            tokenized_docs = [self._tokenize(doc) for doc in self.bm25_docs]

            # 验证分词结果有效性
            if len(tokenized_docs) == 0 or all(len(d) == 0 for d in tokenized_docs):
                raise ValueError("文档分词后为空，请检查分词逻辑")

            # 初始化BM25模型
            self.bm25 = BM25Okapi(tokenized_docs)

            logger.info(f"✅ BM25初始化完成，文档数：{len(self.bm25_docs)}")
        except Exception as e:
            logger.error(f"❌ BM25初始化失败: {str(e)}")
            raise RuntimeError(f"BM25初始化失败: {str(e)}")

    def _hybrid_retrieve(self, question: str) -> List[Dict[str, Any]]:
        """混合检索流程（向量+BM25）

        :param question: 用户问题
        :return: 包含文档和检索信息的字典列表
        """
        results = []

        # 向量检索部分
        vector_results = self.vector_store.similarity_search_with_score(
            question, k=self.config.vector_top_k  # 获取top k结果
        )
        for doc, score in vector_results:
            # 将分数转换为标准余弦值（0~1范围）
            norm_score = (score + 1) / 2  # 如果原始范围是[-1,1]
            results.append({
                "doc": doc,
                "score": norm_score,  # 使用归一化后的分数
                "type": "vector",
                "source": doc.metadata.get("source", "unknown")
            })
            logger.info(f"🔍 向量检索结果: {doc.metadata['source']} - 分数: {score:.4f}")


        # BM25检索部分
        tokenized_query = self._tokenize(question)  # 问题分词
        bm25_scores = self.bm25.get_scores(tokenized_query)  # 计算BM25分数
        # 获取top k的索引（倒序排列）
        top_bm25_indices = np.argsort(bm25_scores)[-self.config.bm25_top_k:][::-1]

        for idx in top_bm25_indices:
            doc = Document(
                page_content=self.bm25_docs[idx],
                metadata=self.doc_metadata[idx]
            )
            results.append({
                "doc": doc,
                "score": float(bm25_scores[idx]),
                "type": "bm25",
                "source": doc.metadata.get("source", "unknown")
            })
            logger.info(f"🔍 BM25检索结果: {doc.metadata['source']} - 分数: {bm25_scores[idx]:.4f}")

        logger.info(f"📚 混合检索后得到{len(results)}篇文档")
        return results

    def _safe_normalize(self,scores: List[float]) -> List[float]:
        """安全归一化处理"""
        if len(scores) == 0:
            return []

        min_val = min(scores)
        max_val = max(scores)

        # 处理常数情况
        if max_val == min_val:
            return [0.5] * len(scores)  # 返回中性值

        return [(x - min_val) / (max_val - min_val) for x in scores]

    def _distribution_aware_normalize(self,scores: List[float], method: str) -> List[float]:
        """分布感知的归一化"""
        if method == "robust":
            # 使用四分位数鲁棒归一化
            q25, q75 = np.percentile(scores, [25, 75])
            iqr = q75 - q25
            if iqr == 0:
                return [(x - q25) for x in scores]
            return [(x - q25) / iqr for x in scores]
        elif method == "zscore":
            # 标准Z-score归一化
            mean = np.mean(scores)
            std = np.std(scores) + 1e-9
            return [(x - mean) / std for x in scores]
        else:
            return self._safe_normalize(scores)

    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        # 按类型分组
        vector_scores = [res["score"] for res in results if res["type"] == "vector"]
        bm25_scores = [res["score"] for res in results if res["type"] == "bm25"]

        # 差异化处理
        vector_norm = self._distribution_aware_normalize(vector_scores, method="zscore")
        bm25_norm = self._distribution_aware_normalize(bm25_scores, method="robust")

        # 合并结果
        idx_vec = 0
        idx_bm25 = 0
        for res in results:
            if res["type"] == "vector":
                res["norm_score"] = vector_norm[idx_vec]
                idx_vec += 1
            else:
                res["norm_score"] = bm25_norm[idx_bm25]
                idx_bm25 += 1
        for res in results:
            # Sigmoid压缩到(0,1)区间
            res["norm_score"] = 1 / (1 + np.exp(-res["norm_score"]))

        for res in results:
            logger.info(
                f"📊 归一化分数: {res['source']} - "
                f"原始分数: {res['score']:.4f} - "
                f"归一化分数: {res['norm_score']:.4f}"
            )
        return results

    def _rerank_documents(self, results: List[Dict], question: str) -> List[Dict]:
        """使用重排序模型优化检索结果

        :param results: 检索结果列表
        :param question: 原始问题
        :return: 重排序后的结果列表
        """
        try:
            # 准备模型输入对（问题-文档）
            pairs = [(question, res["doc"].page_content) for res in results]

            # 对输入进行tokenize和批处理
            inputs = self.rerank_tokenizer(
                pairs,
                padding=True,  # 自动填充
                truncation=True,  # 自动截断
                max_length=512,  # 最大长度限制
                return_tensors="pt"  # 返回PyTorch张量
            )

            # 模型推理
            with torch.no_grad():
                outputs = self.rerank_model(**inputs)
                # 使用sigmoid转换分数
                rerank_scores = torch.sigmoid(outputs.logits).squeeze().tolist()

            # 合并分数
            for res, rerank_score in zip(results, rerank_scores):
                # 加权平均策略
                final_score = (
                        self.config.retrieval_weight * res["norm_score"] +
                        self.config.rerank_weight * rerank_score
                )
                res.update({
                    "rerank_score": rerank_score,
                    "final_score": final_score
                })

            # 按最终分数降序排列
            return sorted(results, key=lambda x: x["final_score"], reverse=True)
        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            return results  # 失败时返回原始排序

    def _retrieve_documents(self, question: str) -> Tuple[List[Document], List[Dict]]:
        """完整检索流程

        :param question: 用户问题
        :return: (文档列表, 分数信息列表)
        """
        try:
            # 混合检索
            raw_results = self._hybrid_retrieve(question)
            if not raw_results:
                return [], []

            # 分数归一化
            norm_results = self._normalize_scores(raw_results)

            # 重排序
            reranked = self._rerank_documents(norm_results, question)

            # 根据阈值过滤结果
            filtered = [
                res for res in reranked
                if res["final_score"] >= self.config.similarity_threshold
            ]
            final_results = sorted(
                filtered,
                key=lambda x: x["final_score"],
                reverse=True
            )[:self.config.final_top_k]
            # 提取文档和分数信息
            docs = [res["doc"] for res in final_results]
            score_info = [{
                "source": res["source"],
                "type": res["type"],
                "vector_score": res.get("score", 0),  # 兼容不同检索类型
                "bm25_score": res.get("score", 0),
                "rerank_score": res["rerank_score"],
                "final_score": res["final_score"]
            } for res in final_results]

            return docs, score_info
        except Exception as e:
            logger.error(f"文档检索失败: {str(e)}")
            raise

    def _build_prompt(self, question: str, context: str) -> str:
        """构建符合deepseek-r1格式的提示模板

        :param question: 用户问题
        :param context: 检索到的上下文
        :return: 格式化后的提示字符串
        """
        if context:
            return (
                "<|im_start|>system\n"
                "你是一位经验丰富的化工安全领域专家...\n"
                "上下文：\n{context}\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "{question}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            ).format(question=question, context=context[:self.config.max_context_length])
        else:
            return (
                "<|im_start|>user\n"
                "{question}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            ).format(question=question)

    def _format_references(self, docs: List[Document], score_info: List[Dict]) -> List[Dict]:
        """格式化参考文档信息"""
        return [
            {
                "file": str(Path(info["source"]).name),  # 文件名
                "content": doc.page_content,  # 截取前500字符
                "score": info["final_score"],  # 综合评分
                "type": info["type"],  # 检索类型
                "full_path": info["source"]  # 完整文件路径
            }
            for doc, info in zip(docs, score_info)
        ]
    def stream_query_model(self, question: str) -> Generator[str, None, None]:
        """纯模型流式生成（不经过RAG）"""
        logger.info(f"🌀 正在直接流式生成: {question[:50]}...")
        try:
            if not question.strip():
                yield "⚠️ 请输入有效问题"
                return

            # 构建基础提示模板（不包含上下文）
            prompt = (
                "<|im_start|>system\n"
                "你是一位经验丰富的化工安全领域专家，请专业且准确地回答问题。\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"{question}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            try:
                full_response = ""
                for chunk in self.llm.stream(prompt):
                    cleaned_chunk = chunk.replace("<|im_end|>", "")
                    if cleaned_chunk:
                        # 发送生成内容（作为普通文本）
                        yield json.dumps({
                            "type": "content",
                            "data": cleaned_chunk
                        }) + "\n"

            except Exception as e:
                logger.error(f"直接生成中断: {str(e)}")
                yield "\n⚠️ 生成过程发生意外中断，请稍后重试"

        except Exception as e:
            logger.exception("直接流式生成错误")
            yield "⚠️ 系统处理请求时发生严重错误，请联系管理员"

    def stream_query_rag_with_kb(self, question: str) -> Generator[str, None, None]:
        """结合知识库的流式RAG生成"""
        logger.info(f"🌊 正在流式处理查询: {question[:50]}...")
        if not question.strip():
            yield "⚠️ 请输入有效问题"
            return

        try:
            # 阶段1：文档检索
            try:
                docs, score_info = self._retrieve_documents(question)
                if not docs:
                    yield "⚠️ 未找到相关文档..."
                    return
            except Exception as e:
                logger.error(f"文档检索失败: {str(e)}")
                yield "⚠️ 文档检索服务暂时不可用"
                return

            # 格式化参考文档信息
            references = self._format_references(docs, score_info)

            # 发送参考文档信息（作为JSON）
            yield json.dumps({
                "type": "references",
                "data": references
            }) + "\n"  # 添加换行符作为结束标记

            # 阶段2：构建上下文
            context = "\n\n".join([
                f"【参考文档{i + 1}】{doc.page_content}\n"
                f"- 来源: {Path(info['source']).name}\n"
                f"- 综合置信度: {info['final_score'] * 100:.1f}%"
                for i, (doc, info) in enumerate(zip(docs, score_info))
            ])

            # 阶段3：构建提示模板
            prompt = self._build_prompt(question, context)

            # 阶段4：流式生成
            try:
                full_response = ""
                for chunk in self.llm.stream(prompt):
                    cleaned_chunk = chunk.replace("<|im_end|>", "")
                    if cleaned_chunk:
                        # 发送生成内容（作为普通文本）
                        yield json.dumps({
                            "type": "content",
                            "data": cleaned_chunk
                        }) + "\n"

            except Exception as e:
                logger.error(f"流式生成中断: {str(e)}")
                yield json.dumps({
                    "type": "error",
                    "data": "\n⚠️ 生成过程发生意外中断"
                }) + "\n"

        except Exception as e:
            logger.exception("流式处理严重错误")
            yield json.dumps({
                "type": "error",
                "data": "⚠️ 系统处理请求时发生严重错误"
            }) + "\n"

