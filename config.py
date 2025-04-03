import torch
from pathlib import Path
from dataclasses import dataclass
from transformers import BitsAndBytesConfig


class Config:
    """RAG系统全局配置类，包含路径、模型参数、硬件设置等配置项"""

    def __init__(self, data_dir: str = "data"):
        # ████████ 路径配置 ████████
        self.data_dir = Path(data_dir)  # 数据存储根目录（自动转换为Path对象）
        self.embedding_model_path = r"C:\Users\coins\Desktop\models\bge-large-zh-v1.5"  # 文本嵌入模型存储路径
        self.vector_db_path = "vector_store/data"  # FAISS向量数据库存储目录
        self.rerank_model_path = r"C:\Users\coins\Desktop\models\bge-reranker-large"  # 重排序模型路径
        self.cache_dir = "cache"  # 缓存目录
        self.max_backups = 5  # 保留的最大备份数量

        # ████████ 硬件配置 ████████
        self.cuda_lazy_init = True  # 延迟CUDA初始化（避免显存立即被占用）
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动检测设备

        # ████████ PDF OCR配置 ████████
        # PDF OCR参数说明:
        # - batch_size: 批处理大小，决定一次处理多少页。值越大速度越快但显存占用越多
        # - min_pages_for_batch: 启用批处理的最小页数，低于此值将逐页处理
        # - det_limit_side_len: 检测分辨率，影响识别精度和速度，值越小速度越快
        # - rec_batch_num: 识别批处理量，值越大速度越快但显存占用越多
        # - det_batch_num: 检测批处理量，值越大速度越快但显存占用越多
        # - use_tensorrt: 是否使用TensorRT加速，需要安装对应版本的TensorRT
        self.pdf_ocr_params = {
            'batch_size': 2,              # 批处理大小
            'min_pages_for_batch': 2,     # 启用批处理的最小页数
            'det_limit_side_len': 640,    # 检测分辨率
            'rec_batch_num': 4,           # 识别批处理量
            'det_batch_num': 2,           # 检测批处理量
            'use_tensorrt': False         # 是否使用TensorRT加速
        }
        
        # 针对1050Ti特别优化的参数（如果需要可以使用）
        self.pdf_ocr_1050ti_params = {
            'batch_size': 2,              # 批处理大小
            'min_pages_for_batch': 2,     # 启用批处理的最小页数
            'det_limit_side_len': 640,    # 检测分辨率
            'rec_batch_num': 4,           # 识别批处理量
            'det_batch_num': 2,           # 检测批处理量
            'use_tensorrt': False         # 1050Ti通常不支持高版本TensorRT
        }
        
        # 针对大文档的优化参数
        self.pdf_ocr_large_doc_params = {
            'batch_size': 1,              # 单页批处理以节省内存
            'min_pages_for_batch': 5,     # 仍然启用批处理，但每批仅处理一页
            'det_limit_side_len': 640,    # 降低检测分辨率
            'rec_batch_num': 2,           # 降低识别批处理量
            'det_batch_num': 1            # 降低检测批处理量
        }

        # ████████ 批处理配置 ████████
        self.batch_size = 32 if torch.cuda.is_available() else 8  # 根据GPU显存自动调整批次大小
        self.normalize_embeddings = True  # 是否对嵌入向量做L2归一化处理

        # ████████ 文本分块配置 ████████
        self.chunk_size = 512  # 文档分块长度（字符数）
        self.chunk_overlap = 128  # 分块重叠区域长度（增强上下文连续性）

        # ████████ Ollama大模型配置 ████████
        self.ollama_base_url = "http://localhost:11434"  # Ollama服务地址
        self.llm_max_tokens = 2048  # 生成文本的最大token数限制
        self.llm_temperature = 0.3  # 温度参数（0-1，控制生成随机性）

        # ████████ RAG检索配置 ████████
        self.max_context_length = 5000  # 输入LLM的上下文最大长度（避免过长导致性能下降）
        self.bm25_top_k = 10  # BM25检索返回的候选文档数
        self.vector_top_k = 10  # 向量检索返回的候选文档数
        self.retrieval_weight = 0.4  # 检索阶段分数权重（与重排序阶段的加权比例）
        self.rerank_weight = 0.6  # 重排序阶段分数权重
        self.similarity_threshold = 0.5  # 相似度过滤阈值（低于此值的文档被丢弃）
        self.final_top_k = 6  # 最终返回给大模型的最相关文档数量


# ████████ 短信服务配置 ████████
URL = "https://gyytz.market.alicloudapi.com/sms/smsSend"  # 阿里云短信API端点
APPCODE = 'f9b3648618f849409d2bdd5c0f07f67a'  # 用户身份验证码（需替换为实际值）
SMS_SIGN_ID = "90362f6500af46bb9dadd26ac6e31e11"  # 短信签名ID（控制台获取）
TEMPLATE_ID = "908e94ccf08b4476ba6c876d13f084ad"  # 短信模板ID（对应具体短信内容格式）
SERVER_URL = 'http://localhost:8000'  # 后端服务地址（生产环境需改为公网域名）