import time
from typing import List, Dict, Optional, Tuple
import fitz
import cv2
import numpy as np
from langchain_core.documents import Document
from paddleocr import PaddleOCR
from multiprocessing import Pool, cpu_count
import os
import logging
import torch  # 添加torch导入以检测GPU
import paddle  # 直接导入paddle检查环境

"""
使用示例:

# 默认方式（如果有GPU会自动使用）
processor = PDFProcessor(lang='ch', use_gpu=True)
docs = processor.process_pdf('example.pdf')

# 手动配置GPU参数
processor = PDFProcessor(lang='ch', use_gpu=True)
processor.configure_gpu(
    batch_size=4,              # 批处理大小
    min_pages_for_batch=3,     # 最小启用批处理的页数
    det_limit_side_len=1280,   # 检测分辨率（更高分辨率可能提高准确性但降低速度）
    rec_batch_num=15,          # 识别批处理量
    det_batch_num=8,           # 检测批处理量
    use_tensorrt=True          # 使用TensorRT加速（需要安装TensorRT）
)
docs = processor.process_pdf('example.pdf')

# 禁用GPU
processor = PDFProcessor(lang='ch', use_gpu=False)
docs = processor.process_pdf('example.pdf')
"""

class PDFProcessor:
    def __init__(self, lang: str = 'ch', use_gpu: bool = True):
        self.lang = lang
        self.use_gpu = use_gpu
        self.base_zoom = 1.2
        self._ocr_engine = None
        self.processes = 1  # 减少进程数以降低CPU负载
        
        # GPU相关参数
        self.gpu_params = {
            'batch_size': 3,          # 批处理大小
            'min_pages_for_batch': 3, # 启用批处理的最小页数
            'det_limit_side_len': 960,# 检测分辨率
            'rec_batch_num': 8,       # 识别批处理量
            'det_batch_num': 4,       # 检测批处理量
            'use_tensorrt': False     # 是否使用TensorRT加速
        }
        
        # 检查GPU可用性
        self.gpu_available = False
        self._check_gpu_availability()
            
    def _check_gpu_availability(self):
        """检查GPU可用性"""
        try:
            # 首先检查环境变量
            if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                
            # 尝试使用torch检测CUDA可用性
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                self.gpu_available = True
                logging.info(f"✅ GPU可用: {gpu_name}, 显存: {gpu_mem:.1f}GB")
                
                # 针对1050Ti特别优化参数
                if "1050 Ti" in gpu_name:
                    logging.info(f"⚡ 检测到1050Ti GPU，应用优化参数")
                    # 1050Ti有4GB显存，针对性设置参数
                    self.gpu_params = {
                        'batch_size': 2,          # 较小批处理大小避免显存溢出
                        'min_pages_for_batch': 2, # 更低的批处理启用阈值
                        'det_limit_side_len': 640,# 降低检测分辨率以减少显存占用
                        'rec_batch_num': 4,       # 较小识别批处理量
                        'det_batch_num': 2,       # 较小检测批处理量
                        'use_tensorrt': False     # 1050Ti通常不支持高版本TensorRT
                    }
            else:
                logging.warning("⚠️ 未检测到可用GPU，将使用CPU模式")
                self.gpu_available = False
                self.use_gpu = False
                
            # 设置paddle环境（无论是否有GPU）
            try:
                if self.use_gpu and self.gpu_available:
                    paddle.set_device('gpu:0')
                    logging.info("✅ Paddle已设置为GPU模式")
                else:
                    paddle.set_device('cpu')
                    logging.info("✅ Paddle已设置为CPU模式")
            except Exception as e:
                logging.warning(f"⚠️ Paddle设备设置失败: {str(e)}")
                if self.use_gpu:
                    self.use_gpu = False
                    logging.warning("⚠️ 已自动切换到CPU模式")
                
        except Exception as e:
            logging.error(f"⚠️ GPU检测失败: {str(e)}")
            self.gpu_available = False
            self.use_gpu = False

    def configure_gpu(self, **kwargs):
        """配置GPU相关参数
        
        参数:
            batch_size (int): 批处理大小
            min_pages_for_batch (int): 启用批处理的最小页数
            det_limit_side_len (int): 检测分辨率
            rec_batch_num (int): 识别批处理量
            det_batch_num (int): 检测批处理量
            use_tensorrt (bool): 是否使用TensorRT加速
        """
        # 更新GPU参数
        for key, value in kwargs.items():
            if key in self.gpu_params:
                self.gpu_params[key] = value
                logging.info(f"更新GPU参数: {key} = {value}")
        
        # 如果已经初始化了OCR引擎，则需要重新初始化
        if self._ocr_engine is not None:
            logging.info("参数已更改，重新初始化OCR引擎...")
            self._ocr_engine = None

    @property
    def ocr_engine(self):
        if self._ocr_engine is None:
            # 根据实际GPU可用性设置use_gpu
            actual_use_gpu = self.use_gpu and self.gpu_available
            
            try:
                # 初始化OCR引擎
                self._ocr_engine = PaddleOCR(
                    use_angle_cls=False,
                    lang=self.lang,
                    use_gpu=actual_use_gpu,  # 使用实际GPU可用性
                    show_log=False,  # 减少日志输出
                    rec_image_shape="3, 48, 320",
                    drop_score=0.6,
                    det_limit_side_len=self.gpu_params['det_limit_side_len'],  # 检测分辨率
                    det_db_unclip_ratio=1.5,  # 优化检测参数
                    rec_algorithm='SVTR_LCNet',
                    rec_batch_num=self.gpu_params['rec_batch_num'],  # 识别批处理量
                    det_batch_num=self.gpu_params['det_batch_num'],  # 检测批处理量
                    use_tensorrt=self.gpu_params['use_tensorrt']  # TensorRT加速
                )
                
                # 日志输出OCR引擎配置情况
                mode_str = "GPU" if actual_use_gpu else "CPU"
                logging.info(f"🔧 OCR引擎初始化完成，使用{mode_str}模式")
            except Exception as e:
                logging.error(f"❌ OCR引擎初始化失败: {str(e)}")
                logging.info("⚠️ 尝试使用CPU模式初始化OCR引擎")
                try:
                    # 确保使用CPU设备
                    paddle.set_device('cpu')
                    self._ocr_engine = PaddleOCR(
                        use_angle_cls=False,
                        lang=self.lang,
                        use_gpu=False,  # 强制使用CPU
                        show_log=False
                    )
                    logging.info("🔧 OCR引擎(CPU模式)初始化完成")
                except Exception as e2:
                    logging.error(f"❌ CPU模式初始化也失败: {str(e2)}")
                    raise e2
            
        return self._ocr_engine

    def _convert_pages(self, pdf_path: str) -> List[Tuple[int, np.ndarray]]:
        try:
            with fitz.open(pdf_path) as doc:
                page_count = doc.page_count
                args = [(pdf_path, pg, self.base_zoom) for pg in range(page_count)]

                with Pool(processes=self.processes) as pool:
                    results = pool.imap(self._convert_page, args)
                    converted = []
                    for result in results:
                        if result is not None:
                            converted.append(result)
                    converted.sort(key=lambda x: x[0])
                    return converted
        except Exception as e:
            logging.error(f"PDF conversion failed: {e}")
            return []

    @staticmethod
    def _convert_page(args: tuple) -> Optional[Tuple[int, np.ndarray]]:
        pdf_path, pg, zoom = args
        try:
            with fitz.open(pdf_path) as doc:
                page = doc[pg]
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                return (pg, img_array)  # 直接返回RGB格式
        except Exception as e:
            logging.warning(f"Page {pg} conversion failed: {e}")
            return None

    def _parse_ocr_result(self, result: list) -> str:
        return "\n".join(
            line[1][0].strip() for line in result[0]
            if line[1][0].strip()
        ) if result and result[0] else ""

    def _batch_process_pages(self, converted_pages: List[Tuple[int, np.ndarray]]) -> List[Document]:
        """使用批处理方式处理页面，提高GPU利用率"""
        documents = []
        batch_size = self.gpu_params['batch_size']
        total_pages = len(converted_pages)
        success_count = 0
        fail_pages = []
        stage_start = time.time()
        
        logging.info(f"📊 使用GPU批处理模式, 批次大小: {batch_size}")
        
        # 确保GPU模式激活
        if not (self.use_gpu and self.gpu_available):
            logging.warning("⚠️ 批处理需要GPU支持，但GPU不可用，将切换到单页处理模式")
            # 单页处理模式
            return self.process_pdf_single_page(converted_pages)
        
        # 首先清理GPU缓存
        try:
            torch.cuda.empty_cache()
        except:
            pass
            
        # 分批处理
        for batch_idx in range(0, total_pages, batch_size):
            batch_end = min(batch_idx + batch_size, total_pages)
            batch_pages = converted_pages[batch_idx:batch_end]
            
            # 显示进度条
            progress = batch_end / total_pages * 100
            self._print_progress_bar(progress, f"{batch_idx+1}-{batch_end}")
            
            # 批量处理前释放内存
            if self.gpu_available and batch_idx > 0:
                # 优化显存使用
                try:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    pass
            
            # 批量处理
            batch_start = time.time()
            
            for pg, img in batch_pages:
                page_num = pg + 1
                try:
                    # 处理前调整图像大小以节省显存
                    if max(img.shape[0], img.shape[1]) > 1600:
                        scale = 1600 / max(img.shape[0], img.shape[1])
                        new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        
                    # OCR处理
                    result = self.ocr_engine.ocr(img, cls=False)
                    page_text = self._parse_ocr_result(result)
                    
                    # 记录文档
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "page": page_num,
                                "image_size": img.shape,
                                "batch_process": True
                            }
                        )
                    )
                    success_count += 1
                except Exception as e:
                    fail_pages.append(page_num)
                    logging.warning(f"  页面 {page_num} 识别失败: {str(e)}")
            
            # 批次处理完成
            batch_time = time.time() - batch_start
            logging.info(f"  批次 {batch_idx//batch_size + 1} 完成，处理 {len(batch_pages)} 页，耗时 {batch_time:.1f}s")
            
            # 每批次后清理GPU内存
            if self.gpu_available:
                try:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    pass
        
        # 最终统计
        total_time = time.time() - stage_start
        # 清空进度条
        print()
        self._print_summary(total_pages, success_count, fail_pages, total_time)
        
        # 按页码排序返回
        documents.sort(key=lambda doc: doc.metadata["page"])
        return documents
        
    def process_pdf_single_page(self, converted_pages: List[Tuple[int, np.ndarray]]) -> List[Document]:
        """使用单页处理模式处理PDF"""
        documents = []
        total_pages = len(converted_pages)
        success_count = 0
        fail_pages = []
        stage_start = time.time()
        
        logging.info("🔍 使用单页处理模式")
        
        for idx, (pg, img) in enumerate(converted_pages):
            page_num = pg + 1
            page_start = time.time()

            try:
                # 显示进度条
                progress = (idx + 1) / total_pages * 100
                self._print_progress_bar(progress, page_num)
                
                # 处理前调整图像大小以节省显存
                if max(img.shape[0], img.shape[1]) > 1600:
                    scale = 1600 / max(img.shape[0], img.shape[1])
                    new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # OCR处理
                result = self.ocr_engine.ocr(img, cls=False)
                page_text = self._parse_ocr_result(result)

                # 记录文档
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "page": page_num,
                            "image_size": img.shape,
                            "process_time": time.time() - page_start
                        }
                    )
                )
                success_count += 1
            except Exception as e:
                fail_pages.append(page_num)
                logging.warning(f"  页面 {page_num} 识别失败: {str(e)}")

        # 最终统计
        total_time = time.time() - stage_start
        # 清空进度条
        print()
        self._print_summary(total_pages, success_count, fail_pages, total_time)
        return documents
        
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """处理PDF并显示实时进度"""
        documents = []
        stage_start = time.time()

        try:
            # 输出处理模式信息
            mode_str = "GPU" if (self.use_gpu and self.gpu_available) else "CPU"
            logging.info(f"🚀 开始处理PDF文件，使用{mode_str}模式")
            
            # 阶段1：PDF转图像
            logging.info("▌正在解析PDF页面...")
            converted_pages = self._convert_pages(pdf_path)
            if not converted_pages:
                logging.warning("⚠️ 未找到可处理页面")
                return []

            total_pages = len(converted_pages)
            parse_time = time.time() - stage_start
            logging.info(f"✅ 页面解析完成，共 {total_pages} 页（耗时 {parse_time:.1f}s）\n")

            # 阶段2：OCR处理
            logging.info("▌开始文字识别处理:")

            # GPU模式下使用批处理提高性能
            if self.use_gpu and self.gpu_available and total_pages > self.gpu_params['min_pages_for_batch']:
                return self._batch_process_pages(converted_pages)
            else:
                # 单页处理模式
                return self.process_pdf_single_page(converted_pages)

        except Exception as e:
            logging.error(f"💥 处理流程异常终止: {str(e)}")
            return []

    def _print_progress_bar(self, progress: float, current_page: int):
        """控制台进度条显示，改为单行固定格式"""
        bar_length = 20
        filled = int(progress / 100 * bar_length)
        bar = '█' * filled + '░' * (bar_length - filled)
        # 使用\r确保每个文件只占用一行
        print(f"\r处理中 {bar} {progress:.1f}% | 页面 {current_page}", end='', flush=True)

    def _print_summary(self, total: int, success: int, failed: list, duration: float):
        """输出处理结果摘要"""
        print("\n\n" + "═" * 50)
        logging.info(f"📊 处理统计:")
        logging.info(f"  总页数    : {total} 页")
        logging.info(f"  成功识别  : {success} 页 ({success / total * 100:.1f}%)")
        if failed:
            logging.info(f"  失败页面  : {', '.join(map(str, failed))}")
        logging.info(f"  总耗时    : {duration:.1f} 秒")
        logging.info(f"  平均速度  : {duration / total:.1f} 秒/页" if total > 0 else "")
        print("═" * 50 + "\n")