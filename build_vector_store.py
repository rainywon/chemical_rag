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
from typing import List, Dict, Optional, Set, Tuple  # 导入类型提示
import logging  # 导入日志模块，用于记录运行日志
from concurrent.futures import ThreadPoolExecutor, as_completed  # 导入线程池模块，支持并行加载PDF文件
from tqdm import tqdm  # 导入进度条模块，用于显示加载进度
from config import Config  # 导入配置类，用于加载配置参数
import shutil  # 用于文件操作

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
        self.chunk_cache_path = Path(self.config.vector_db_path) / "chunks_cache.json"
        self.failed_files_count = 0  # 添加未成功加载文件计数器
        self.need_rebuild_index = False  # 标记是否需要重建向量索引

    def _load_processing_state(self) -> Dict:
        """加载之前的文件处理状态"""
        if self.state_file.exists():  # 如果状态文件存在，则加载
            with open(self.state_file, "r", encoding='utf-8') as f:
                return json.load(f)  # 返回加载的状态文件内容
        return {}  # 如果文件不存在，返回空字典

    def _save_processing_state(self):
        """保存当前文件处理状态"""
        with open(self.state_file, "w", encoding='utf-8') as f:
            json.dump(self.processed_files, f, indent=2, ensure_ascii=False)  # 保存已处理文件的信息

    def _should_process(self, file_path: Path) -> bool:
        """判断文件是否需要处理"""
        file_stat = file_path.stat()  # 获取文件状态
        file_info = {
            "path": str(file_path),  # 文件路径
            "size": file_stat.st_size,  # 文件大小
            "mtime": file_stat.st_mtime  # 文件修改时间
        }

        existing = self.processed_files.get(str(file_path))  # 查找已处理的文件记录
        if not existing:  # 如果文件未处理过，则需要处理
            return True
        return not (existing["size"] == file_info["size"]
                    and existing["mtime"] == file_info["mtime"])  # 如果文件修改时间或大小不一致，则需要重新处理

    def _check_file_changes(self, current_files: List[Path]) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        检查文件变化，返回需要添加、更新和删除的文件
        
        Args:
            current_files: 当前文件系统中的文件列表
            
        Returns:
            Tuple[Set[str], Set[str], Set[str]]: 新增文件、更新文件和删除文件的路径集合
        """
        # 获取当前文件系统中的所有文件路径
        current_file_paths = {str(f) for f in current_files}
        
        # 获取已处理文件的路径
        processed_file_paths = set(self.processed_files.keys())
        
        # 需要新增的文件（当前存在但未处理过）
        new_files = current_file_paths - processed_file_paths
        
        # 需要更新的文件（已处理过但需要重新处理）
        update_files = {str(f) for f in current_files if self._should_process(f)}
        
        # 需要删除的文件（已处理过但当前不存在）
        deleted_files = processed_file_paths - current_file_paths
        
        return new_files, update_files, deleted_files

    def _load_single_document(self, file_path: Path) -> Optional[List[Dict]]:
        """多线程加载单个文档文件（支持 PDF、DOCX、DOC）"""
        try:
            if not self._should_process(file_path):
                logger.info(f"⏭ 跳过未修改文件: {file_path.name}")
                return None

            file_extension = file_path.suffix.lower()
            docs = []

            if file_extension == ".pdf":
                try:
                    # 检查PDF页数
                    import fitz
                    with fitz.open(str(file_path)) as doc:
                        page_count = doc.page_count
                        logger.info(f"PDF文件 '{file_path.name}' 共有 {page_count} 页")
                        
                    # 如果页数很多，优化内存设置
                    processor = PDFProcessor(lang='ch', use_gpu=True)
                    
                    # 针对页数较多的PDF进行特别优化
                    if page_count > 30:
                        logger.info(f"PDF页数较多({page_count}页)，应用内存优化配置")
                        processor.configure_gpu(
                            batch_size=1,               # 单页批处理以节省内存
                            min_pages_for_batch=5,      # 仍然启用批处理，但每批仅处理一页
                            det_limit_side_len=640,     # 降低检测分辨率
                            rec_batch_num=2,            # 降低识别批处理量
                            det_batch_num=1             # 降低检测批处理量
                        )
                    else:
                        # 针对1050Ti优化
                        processor.configure_gpu(
                            batch_size=2,              # 较小批处理大小避免显存溢出
                            min_pages_for_batch=2,     # 更低的批处理启用阈值
                            det_limit_side_len=640,    # 降低检测分辨率以减少显存占用
                            rec_batch_num=4,           # 较小识别批处理量
                            det_batch_num=2            # 较小检测批处理量
                        )
                    
                    # 处理PDF
                    docs = processor.process_pdf(str(file_path))
                    
                    # 检查处理结果
                    if docs and len(docs) < page_count * 0.5:
                        logger.warning(f"⚠️ 只识别出 {len(docs)}/{page_count} 页，低于50%，可能有问题")
                    elif docs:
                        logger.info(f"✅ 成功识别 {len(docs)}/{page_count} 页")
                    
                    # 处理后清理内存
                    import gc
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
                        
                except Exception as e:
                    logger.error(f"处理PDF文件 '{file_path.name}' 失败: {str(e)}")
                    return None
                    
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
                    "pages": len(docs),
                    "file_name": file_path.name  # 添加文件名，便于后续查找
                }
                # 统一添加元数据
                for doc in docs:
                    doc.metadata["source"] = str(file_path)
                    doc.metadata["file_name"] = file_path.name
                return docs

        except Exception as e:
            logger.error(f"加载 {file_path} 失败: {str(e)}", exc_info=True)
            self.failed_files_count += 1
            return None

    def _cleanup_deleted_files(self):
        """
        清理已删除文件的缓存信息
        """
        if not self.chunk_cache_path.exists():
            return
            
        try:
            # 获取当前所有文件路径
            current_files = []
            subfolder_list = ['标准']  # 可以从配置中获取
            for subfolder in subfolder_list:
                folder_path = self.config.data_dir / subfolder
                if folder_path.exists() and folder_path.is_dir():
                    current_files.extend([f for f in folder_path.rglob("*") 
                                       if f.suffix.lower() in ['.pdf', '.docx', '.doc']])
                    
            # 分析文件变化
            new_files, update_files, deleted_files = self._check_file_changes(current_files)
            
            if not deleted_files and not update_files:
                logger.info("没有检测到文件变化，跳过清理")
                return
                
            # 如果有文件被删除或更新，需要清理缓存
            with open(self.chunk_cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            # 清理已删除文件和更新文件的数据
            files_to_clean = deleted_files.union(update_files)
            if files_to_clean:
                logger.info(f"检测到 {len(deleted_files)} 个已删除文件, {len(update_files)} 个已更新文件")
                
                # 从处理状态中删除已删除的文件
                for file_path in deleted_files:
                    if file_path in self.processed_files:
                        del self.processed_files[file_path]
                        logger.info(f"从处理状态中移除已删除文件: {Path(file_path).name}")
                
                # 从缓存中过滤掉已删除和已更新的文件块
                clean_chunks = []
                removed_count = 0
                
                for chunk in cache_data.get("chunks", []):
                    chunk_source = chunk.get("metadata", {}).get("source", "")
                    if chunk_source in files_to_clean:
                        removed_count += 1
                        continue
                    clean_chunks.append(chunk)
                
                # 更新缓存数据
                if removed_count > 0:
                    cache_data["chunks"] = clean_chunks
                    cache_data["file_state"] = self.processed_files
                    with open(self.chunk_cache_path, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"✅ 已从缓存中移除 {removed_count} 个过时的文本块")
                    self.need_rebuild_index = True
                    
                    # 保存更新后的处理状态
                    self._save_processing_state()
                    
        except Exception as e:
            logger.error(f"清理缓存失败: {str(e)}")
            # 遇到错误，为安全起见标记需要重建索引
            self.need_rebuild_index = True

    def load_documents(self) -> List:
        """并行加载所有文档"""
        logger.info("⌛ 开始加载文档...")
        
        # 首先清理已删除文件的缓存
        self._cleanup_deleted_files()

        # 获取 data_dir 下的所有子文件夹
        subfolders = ['标准性文件','法律', '规范性文件']  # '标准性文件','法律', '规范性文件'
        document_files = []

        # 遍历每个子文件夹，获取其中的文档文件
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
        
        # 过滤出需要处理的文件
        files_to_process = [f for f in document_files if self._should_process(f)]
        logger.info(f"其中 {len(files_to_process)} 个文件需要处理")
        
        if not files_to_process:
            logger.info("没有新文件需要处理，使用现有缓存")
            # 从缓存加载文档
            return self._load_documents_from_cache()

        # 限制线程池大小以避免资源争用
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(self._load_single_document, file) for file in files_to_process]
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
        
        # 处理文件变化后，将新处理的文档与现有缓存合并
        if results and self.chunk_cache_path.exists():
            self._merge_with_cache(results)
            # 设置标记，表示需要重建索引
            self.need_rebuild_index = True
            return self._load_documents_from_cache()
            
        return results

    def _load_documents_from_cache(self) -> List[Document]:
        """从缓存加载文档"""
        try:
            if self.chunk_cache_path.exists():
                with open(self.chunk_cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    chunks = [
                        Document(
                            page_content=chunk["content"],
                            metadata=chunk["metadata"]
                        )
                        for chunk in cache_data.get("chunks", [])
                    ]
                    logger.info(f"从缓存加载了 {len(chunks)} 个文本块")
                    return chunks
        except Exception as e:
            logger.error(f"加载缓存失败: {str(e)}")
        
        return []

    def _merge_with_cache(self, new_docs: List[Document]):
        """将新处理的文档与现有缓存合并"""
        try:
            # 加载现有缓存
            if not self.chunk_cache_path.exists():
                logger.info("缓存不存在，跳过合并")
                return
                
            with open(self.chunk_cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            # 使用与process_files相同的分块逻辑处理新文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""],
                length_function=len,
                add_start_index=True,
                is_separator_regex=False
            )
            
            # 处理新文档
            new_chunks = []
            for doc in new_docs:
                metadata = doc.metadata.copy()
                split_texts = text_splitter.split_text(doc.page_content)
                for text in split_texts:
                    # 生成内容哈希
                    content_hash = hashlib.md5(text.encode()).hexdigest()
                    enhanced_metadata = metadata.copy()
                    enhanced_metadata["content_hash"] = content_hash
                    new_chunks.append({
                        "content": text,
                        "metadata": enhanced_metadata
                    })
                    
            # 合并缓存
            existing_chunks = cache_data.get("chunks", [])
            
            # 从现有缓存中过滤掉与新文档相同来源的块
            filtered_chunks = []
            new_sources = {doc.metadata.get("source") for doc in new_docs}
            for chunk in existing_chunks:
                chunk_source = chunk.get("metadata", {}).get("source", "")
                if chunk_source not in new_sources:
                    filtered_chunks.append(chunk)
                    
            # 合并过滤后的现有块和新块
            merged_chunks = filtered_chunks + new_chunks
            
            # 更新缓存
            cache_data["chunks"] = merged_chunks
            cache_data["file_state"] = self.processed_files
            
            with open(self.chunk_cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"✅ 已将 {len(new_chunks)} 个新块合并到缓存中，总计 {len(merged_chunks)} 个块")
            
        except Exception as e:
            logger.error(f"合并缓存失败: {str(e)}")

    def process_files(self) -> List:
        """优化的文件处理流程"""
        logger.info("开始文件处理流程")
        
        # 首先检查是否有现有缓存可用
        if self.chunk_cache_path.exists() and not self.need_rebuild_index:
            try:
                with open(self.chunk_cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                    # 比较缓存中的文件状态与当前处理状态
                    if cache_data.get("file_state", {}) == self.processed_files:
                        chunks = [
                            Document(
                                page_content=chunk["content"],
                                metadata=chunk["metadata"]
                            )
                            for chunk in cache_data.get("chunks", [])
                        ]
                        logger.info(f"✅ 从缓存加载 {len(chunks)} 个分块")
                        return chunks
            except Exception as e:
                logger.error(f"缓存加载失败: {str(e)}")

        # 如果没有合适的缓存，处理所有文档
        all_docs = self.load_documents()

        if not all_docs:
            logger.warning("没有可处理的文件内容")
            return []

        # 只有在没有从load_documents中使用缓存时才需要进行分块处理
        if not self.need_rebuild_index:
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

            with open(self.chunk_cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 分块缓存已保存至 {self.chunk_cache_path}")
            return chunks  # 直接返回包含原始元数据的分块
        else:
            # 如果已经通过load_documents加载并更新了缓存，直接返回文档
            return all_docs

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

    def backup_vector_db(self):
        """备份现有向量数据库"""
        vector_db_path = Path(self.config.vector_db_path)
        if not vector_db_path.exists():
            return False
            
        try:
            # 创建备份目录
            backup_dir = vector_db_path.parent / f"{vector_db_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制所有文件到备份目录
            for item in vector_db_path.glob('*'):
                if item.is_file():
                    shutil.copy2(item, backup_dir)
                elif item.is_dir():
                    shutil.copytree(item, backup_dir / item.name)
                    
            logger.info(f"✅ 向量数据库已备份至 {backup_dir}")
            return True
        except Exception as e:
            logger.error(f"备份向量数据库失败: {str(e)}")
            return False

    def build_vector_store(self):
        """构建向量数据库"""
        logger.info("开始构建向量数据库")

        # 创建必要目录
        Path(self.config.vector_db_path).mkdir(parents=True, exist_ok=True)

        # 处理文档
        chunks = self.process_files()  # 处理文档并分块
        
        if not chunks:
            logger.warning("没有文档块可以处理，跳过向量存储构建")
            return

        # 检查是否需要重建索引
        if self.need_rebuild_index:
            logger.info("检测到文件变化，需要重建向量索引")
            if Path(self.config.vector_db_path).exists() and any(Path(self.config.vector_db_path).glob('*')):
                self.backup_vector_db()

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
