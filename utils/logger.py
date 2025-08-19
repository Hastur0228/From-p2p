from __future__ import annotations

import logging
from pathlib import Path
import sys
from datetime import datetime


def setup_logger(log_dir: str | Path, name: str = 'train') -> logging.Logger:
    """
    配置日志记录器：同时输出到控制台与文件，UTF-8 编码。
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f"{name}_{time_str}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 若重复创建，避免多重 handler
    if logger.handlers:
        return logger

    # 文件 handler
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(fmt)

    # 控制台 handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"日志文件: {log_path}")
    return logger


