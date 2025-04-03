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

class PDFProcessor:
    def __init__(self, lang: str = 'ch', use_gpu: bool = True):
        self.lang = lang
        self.use_gpu = use_gpu
        self.base_zoom = 1.2
        self._ocr_engine = None
        self.processes = 1  # 减少进程数以降低CPU负载

    @property
    def ocr_engine(self):
        if self._ocr_engine is None:
            self._ocr_engine = PaddleOCR(
                use_angle_cls=False,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,  # 减少日志输出
                rec_image_shape="3, 48, 320",
                drop_score=0.6,
                det_limit_side_len=960,  # 降低检测分辨率
                det_db_unclip_ratio=1.5,  # 优化检测参数
                rec_algorithm='SVTR_LCNet',
                rec_batch_num=10,  # 增大识别批处理量
                det_batch_num=5,   # 增大检测批处理量
                use_tensorrt=False  # 启用TensorRT加速（如果环境支持）
            )
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

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """处理PDF并显示实时进度"""
        documents = []
        stage_start = time.time()

        try:
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
            success_count = 0
            fail_pages = []

            for idx, (pg, img) in enumerate(converted_pages):
                page_num = pg + 1
                page_start = time.time()

                try:
                    # 显示进度条
                    progress = (idx + 1) / total_pages * 100
                    self._print_progress_bar(progress, page_num)

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
            self._print_summary(total_pages, success_count, fail_pages, total_time)
            return documents

        except Exception as e:
            logging.error(f"💥 处理流程异常终止: {str(e)}")
            return []

    def _print_progress_bar(self, progress: float, current_page: int):
        """控制台进度条显示"""
        bar_length = 30
        filled = int(progress / 100 * bar_length)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r▌处理中 {bar} {progress:.1f}% | 正在识别第 {current_page} 页", end='', flush=True)

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