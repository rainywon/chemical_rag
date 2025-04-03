import hashlib
import sys
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 导入文档分割工具
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings  # 导入HuggingFace嵌入模型
from langchain_community.vectorstores import FAISS  # 导入FAISS用于构建向量数据库
from langchain_community.document_loaders import UnstructuredPDFLoader  # 新增导入
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import os
import json
from pathlib import Path  # 导入Path，用于路径处理
from datetime import datetime  # 导入datetime，用于记录时间戳
from typing import List, Dict, Optional  # 导入类型提示
import logging  # 导入日志模块，用于记录运行日志
from concurrent.futures import ThreadPoolExecutor, as_completed  # 导入线程池模块，支持并行加载PDF文件
from tqdm import tqdm  # 导入进度条模块，用于显示加载进度
from config import Config  # 导入配置类，用于加载配置参数
import os

from pdf_cor_extractor.pdf_ocr_extractor import PDFProcessor


# 配置日志格式
# 配置日志格式，指定输出到stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],  # 明确输出到stdout
    force=True  # 关键：强制覆盖现有配置
)
logger = logging.getLogger(__name__)


class VectorDBBuilder:
    def __init__(self, config: Config):
        """
        初始化向量数据库构建器
        Args:
            config (Config): 配置类，包含必要的配置
        """
        self.config = config
        self.state_file = Path(config.vector_db_path) / "processing_state.json"
        self.processed_files: Dict[str, Dict] = self._load_processing_state()  # 加载已处理的文件记录
        self.failed_files_count = 0  # 添加未成功加载文件计数器

    def _load_processing_state(self) -> Dict:
        """加载之前的文件处理状态"""
        if self.state_file.exists():  # 如果状态文件存在，则加载
            with open(self.state_file, "r") as f:
                return json.load(f)  # 返回加载的状态文件内容
        return {}  # 如果文件不存在，返回空字典

    def _save_processing_state(self):
        """保存当前文件处理状态"""
        with open(self.state_file, "w") as f:
            json.dump(self.processed_files, f, indent=2)  # 保存已处理文件的信息

    def _should_process(self, pdf_path: Path) -> bool:
        """判断文件是否需要处理"""
        file_stat = pdf_path.stat()  # 获取文件状态
        file_info = {
            "path": str(pdf_path),  # 文件路径
            "size": file_stat.st_size,  # 文件大小
            "mtime": file_stat.st_mtime  # 文件修改时间
        }

        existing = self.processed_files.get(str(pdf_path))  # 查找已处理的文件记录
        if not existing:  # 如果文件未处理过，则需要处理
            return True
        return not (existing["size"] == file_info["size"]
                    and existing["mtime"] == file_info["mtime"])  # 如果文件修改时间或大小不一致，则需要重新处理

    def _load_single_document(self, file_path: Path) -> Optional[List[Dict]]:
        """多线程加载单个文档文件（支持 PDF、DOCX、DOC）"""
        try:
            if not self._should_process(file_path):
                logger.info(f"⏭ 跳过未修改文件: {file_path.name}")
                return None

            file_extension = file_path.suffix.lower()
            docs = []

            if file_extension == ".pdf":
                processor = PDFProcessor(lang='ch', use_gpu=True)
                # 配置GPU参数，针对1050Ti优化
                processor.configure_gpu(
                    batch_size=2,              # 较小批处理大小避免显存溢出
                    min_pages_for_batch=2,     # 更低的批处理启用阈值
                    det_limit_side_len=640,    # 降低检测分辨率以减少显存占用
                    rec_batch_num=4,           # 较小识别批处理量
                    det_batch_num=2            # 较小检测批处理量
                )
                docs = processor.process_pdf(str(file_path))
            elif file_extension in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(str(file_path))
                docs = loader.load()
            else:
                logger.warning(f"不支持的文件格式: {file_path.name}")
                return None

            if docs:
                # 记录文件处理信息
                self.processed_files[str(file_path)] = {
                    "size": file_path.stat().st_size,
                    "mtime": file_path.stat().st_mtime,
                    "processed_at": datetime.now().isoformat(),
                    "pages": len(docs)
                }
                # 统一添加元数据
                for doc in docs:
                    doc.metadata["source"] = str(file_path)
                return docs

        except Exception as e:
            logger.error(f"加载 {file_path} 失败: {str(e)}", exc_info=True)
            self.failed_files_count += 1
            return None

    def load_documents(self) -> List:
        """并行加载所有文档"""
        logger.info("⌛ 开始加载文档...")

        # 获取 data_dir 下的所有子文件夹
        subfolders = ['标准']  # '标准性文件','法律', '规范性文件'
        document_files = []

        # 遍历每个子文件夹，获取其中的所有 PDF 文件
        for subfolder in subfolders:
            folder_path = self.config.data_dir / subfolder
            if folder_path.exists() and folder_path.is_dir():
                document_files.extend([f for f in folder_path.rglob("*") 
                                    if f.suffix.lower() in ['.pdf', '.docx', '.doc']])
            else:
                logger.warning(f"子文件夹 {subfolder} 不存在或不是目录: {folder_path}")
                
        # 过滤并排序文件（先处理较小的文件，避免大文件占用显存）
        document_files = sorted(document_files, key=lambda x: x.stat().st_size)
        logger.info(f"发现 {len(document_files)} 个待处理文件")

        # 限制线程池大小以避免资源争用
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(self._load_single_document, file) for file in document_files]
            results = []
            with tqdm(total=len(futures), desc="加载文档", unit="files") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        results.extend(res)
                        pbar.update(1)
                        pbar.set_postfix_str(f"已加载 {len(res)} 页")
                    else:
                        pbar.update(1)
        
        # 在处理完成后清理GPU缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
            
        logger.info(f"✅ 成功加载 {len(results)} 页文档")
        logger.info(f"❌ 未成功加载 {self.failed_files_count} 个文件")
        self._save_processing_state()
        return results

    def process_files(self) -> List:
        """优化的文件处理流程"""
        logger.info("开始文件处理流程")
        # 缓存文件路径
        cache_path = Path(self.config.vector_db_path) / "chunks_cache.json"
        chunks = []

        # 尝试加载缓存
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    if cache_data["file_state"] == self.processed_files:
                        chunks = [
                            Document(
                                page_content=chunk["content"],
                                metadata=chunk["metadata"]
                            )
                            for chunk in cache_data["chunks"]
                        ]
                        logger.info(f"✅ 从缓存加载 {len(chunks)} 个分块")
                        return chunks
            except Exception as e:
                logger.error(f"缓存加载失败: {str(e)}")


        all_docs = self.load_documents()

        if not all_docs:
            raise ValueError("没有可处理的文件内容")

        # 优化后的文本分割配置
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,  # 根据中文平均长度调整
            chunk_overlap=self.config.chunk_overlap,  # 适当增加重叠量
            separators=[
                "\n\n",  # 优先按空行分割
                "\n",  # 其次按换行符分割
                "。", "；",  # 中文句末标点
                "！", "？",  # 中文感叹号和问号
                "，",  # 中文逗号（谨慎使用）
                " ",  # 空格
                ""  # 最后按字符分割
            ],
            length_function=len,
            add_start_index=True,
            is_separator_regex=False  # 明确使用字面分隔符
        )

        logger.info("开始智能分块处理...")
        chunks = []
        # 新增哈希生成和元数据增强
        with tqdm(total=len(all_docs), desc="处理文档分块") as pbar:
            for doc in all_docs:
                metadata = doc.metadata.copy()
                split_texts = text_splitter.split_text(doc.page_content)
                for text in split_texts:
                    # 生成内容哈希
                    content_hash = hashlib.md5(text.encode()).hexdigest()
                    enhanced_metadata = metadata.copy()
                    enhanced_metadata["content_hash"] = content_hash

                    chunks.append(Document(
                        page_content=text,
                        metadata=enhanced_metadata
                    ))
                pbar.update(1)
        logger.info(f"生成 {len(chunks)} 个语义连贯的文本块")

        # 保存处理结果到缓存
        cache_data = {
            "file_state": self.processed_files,
            "chunks": [{
                "content": chunk.page_content,
                "metadata": chunk.metadata
            } for chunk in chunks]
        }

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 分块缓存已保存至 {cache_path}")
        return chunks  # 直接返回包含原始元数据的分块

    def create_embeddings(self) -> HuggingFaceEmbeddings:
        """创建嵌入模型实例"""
        logger.info("初始化嵌入模型...")
        return HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_path,  # 嵌入模型的路径
            model_kwargs={"device": self.config.device},  # 设置设备为CPU或GPU
            encode_kwargs={
                "batch_size": self.config.batch_size,  # 批处理大小
                "normalize_embeddings": self.config.normalize_embeddings  # 是否归一化嵌入
            },
        )

    def build_vector_store(self):
        """构建向量数据库"""
        logger.info("开始构建向量数据库")

        # 创建必要目录
        Path(self.config.vector_db_path).mkdir(parents=True, exist_ok=True)

        # 处理文档
        chunks = self.process_files()  # 处理文档并分块

        # 生成嵌入模型
        embeddings = self.create_embeddings()

        # 构建向量存储
        logger.info("生成向量...")
        # 构建向量存储时显式指定
        vector_store = FAISS.from_documents(
            chunks,
            embeddings,
            distance_strategy=DistanceStrategy.COSINE  # 明确指定余弦相似度
        )

        # 保存向量数据库
        vector_store.save_local(str(self.config.vector_db_path))  # 保存向量存储到指定路径
        logger.info(f"向量数据库已保存至 {self.config.vector_db_path}")  # 输出保存路径


if __name__ == "__main__":
    try:
        # 初始化配置
        config = Config()

        # 构建向量数据库
        builder = VectorDBBuilder(config)
        builder.build_vector_store()

    except Exception as e:
        logger.exception("程序运行出错")  # 记录程序异常
    finally:
        logger.info("程序运行结束")  # 程序结束日志
