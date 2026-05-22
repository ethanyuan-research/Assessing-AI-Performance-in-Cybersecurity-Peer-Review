import json
import logging
import os
import re
import pprint
import shlex
import subprocess
import sys
import threading
from collections import deque, Counter, defaultdict
from time import sleep
from pathlib import Path
from tqdm import tqdm

import requests
import config

# 保留原有的配置
pp = pprint.PrettyPrinter(width=80)
SLEEP_INTERVAL = 1

# ... [保留原有的 get_id_list, check_pdf_exist 等辅助函数] ...

def get_id_list():
    id_list = []
    with open(config.ARXIV_JSON_PATH) as f:
        for line in f:
            meta_data = json.loads(line)
            paper_id = meta_data['id']
            for version_entry in meta_data['versions']:
                id_list.append(paper_id + version_entry['version'])
    return id_list

def check_pdf_exist(id_str):
    # 简单的文件名检查逻辑
    pdf_name = id_str.replace('/', '_')
    if not pdf_name.endswith('.pdf'):
        pdf_name += '.pdf'
    pdf_path = config.ARXIV_PDFS_PATH.joinpath(pdf_name)
    return pdf_path.exists(), pdf_path

class Transformer(threading.Thread):
    '''
    Process the existing arxiv papers with pdf2txt.
    '''
    def __init__(self, dq_to_pdf2txt: deque, tried_times: Counter):
        threading.Thread.__init__(self, name="Transformer")
        # 不需要下载队列了
        self.dq_to_pdf2txt = dq_to_pdf2txt
        self.tried_times = tried_times
        self.logger = logging.getLogger()

    def run(self):
        while True:
            try:
                # 非阻塞获取，或者设置超时
                try:
                    item = self.dq_to_pdf2txt.popleft()
                except IndexError:
                    sleep(0.1)
                    # 如果队列空了，且主线程已经分发完毕（通过外部控制或Sentinel），这里简单处理为等待
                    continue

                if item is None: # Sentinel 信号，退出线程
                    break

                id_str, src = item
                basename = os.path.splitext(os.path.basename(src))[0]
                
                # 检查目标文件是否已存在，避免重复转换
                dst = os.path.join(config.ArxivDataset.txt_dir, basename + '.txt')
                # if os.path.exists(dst) and os.path.getsize(dst) > 0:
                #     continue # 跳过已存在的

                info_str = f"PDF2TXT {basename}({len(self.dq_to_pdf2txt)} left): "
                
                # 调用 pdf2txt
                cmd = f'pdf2txt.py "{src}" -o "{dst}"'
                ret = subprocess.run(shlex.split(cmd), stderr=subprocess.PIPE)

                if ret.returncode != 0:
                    self.logger.error(f"{info_str} ERROR, returned {ret.returncode}; info:{ret.stderr}.")
                else:
                    self.logger.info(f"{info_str} Success.")

            except Exception as e:
                self.logger.error(f"Transformer Exception: {e}")

# ... [保留原有的 _dblp_parse 等函数] ...
def _dblp_parse():
    # (保持您原有的代码不变，这里省略以节省篇幅)
    print("Selecting negative samples by parsing dblp results and arxiv meta data.")
    result_file = config.DBLP_RESULT_PATH
    with open(result_file, 'r') as f:
        dblp_res = json.load(f)
    if config.ArxivDataset.to_exclude_json.exists():
        with open(config.ArxivDataset.to_exclude_json, 'r') as f:
            to_exclude = set(json.load(f))
    else:
        to_exclude = set()
        
    result_by_rules = defaultdict(list)

    with open(config.ARXIV_JSON_PATH, 'r') as f:
        meta_data_all = []
        for i, line in enumerate(f):
            meta_data_all.append(json.loads(line))

    dblp_map = {d['id']: d for d in dblp_res}

    for meta_data in meta_data_all:
        if meta_data['id'] not in dblp_map:
            continue
        # ... (这里省略具体的过滤逻辑，保持您原代码逻辑即可) ...
        # 简化版：只要在dblp_map里就加进去，或者您保留原有的复杂逻辑
        # 为了演示，这里假设您已经有了正确的 filter_list
        pass 
    
    # 注意：如果您不需要重新生成 filter_json，这部分可以跳过
    # 直接读取现有的 filter_json 即可
    return {} 

def _convert_only_pipeline():
    """
    只转换不下载的 Pipeline
    """
    logs_folder = "2023_Mar"
    logs_date = "ConvertOnly"

    # Ensure log directory exists
    log_dir = Path(f'./logs/{logs_folder}')
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    # 清除旧的 handler 防止重复
    logger.handlers = []
    
    file_handler = logging.FileHandler(f'./logs/{logs_folder}/convert_pdf2txt_{logs_date}.log')
    file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)s %(message)s'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)s %(message)s'))
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    # 1. 获取需要处理的 ID 列表
    if config.ArxivDataset.filter_json.exists():
        print(f"Loading ID list from {config.ArxivDataset.filter_json}")
        with open(config.ArxivDataset.filter_json, 'r') as f:
            id_list = json.load(f)
    else:
        print("Filter list not found, falling back to full list!")
        id_list = get_id_list()

    dq_to_pdf2txt = deque()
    tried_times = Counter()

    print(f"Checking local PDFs for {len(id_list)} papers...")
    
    # 2. 预先填充转换队列
    count_found = 0
    count_missing = 0
    
    for id_str in tqdm(id_list, desc="Checking PDFs"):
        # 尝试不同的后缀组合找到本地文件
        # 您的下载逻辑可能去掉了 v 版本号，也可能保留了，这里需要兼容检查
        
        # 逻辑1: 直接替换 / 为 _ 并加 .pdf
        slug_v = id_str.replace('/', '_')
        if not slug_v.endswith('.pdf'): slug_v += '.pdf'
        path_v = config.ARXIV_PDFS_PATH.joinpath(slug_v)

        # 逻辑2: 去掉版本号 (例如 2105.0001v1 -> 2105.0001.pdf)
        slug_no_v = re.sub(r'v\d+\.pdf$', '.pdf', slug_v)
        path_no_v = config.ARXIV_PDFS_PATH.joinpath(slug_no_v)
        
        final_path = None
        if path_v.exists():
            final_path = path_v
        elif path_no_v.exists():
            final_path = path_no_v
            
        if final_path:
            dq_to_pdf2txt.append((id_str, str(final_path)))
            count_found += 1
        else:
            # logger.warning(f"PDF missing for {id_str}")
            count_missing += 1

    print(f"\nFound {count_found} PDFs. Missing {count_missing} PDFs.")
    print(f"Starting conversion with 15 Transformer threads...")

    # 3. 启动转换线程 (只转换，不下载)
    # 增加线程数，因为只是本地IO和CPU计算
    num_threads = 15 
    transformers = [Transformer(dq_to_pdf2txt, tried_times) for _ in range(num_threads)]

    for t in transformers:
        t.start()

    # 4. 等待队列处理完毕
    # 这里我们不能 join transformers，因为它们是 while True 循环
    # 我们需要监控队列是否为空，或者发送 Sentinel
    
    # 发送结束信号
    for _ in range(num_threads):
        dq_to_pdf2txt.append(None)

    for t in transformers:
        t.join()

    print("All conversion tasks finished.")


if __name__ == "__main__":
    # 确保 exclude json 存在，避免报错
    if not config.ArxivDataset.to_exclude_json.parent.exists():
        config.ArxivDataset.to_exclude_json.parent.mkdir(parents=True, exist_ok=True)
    if not config.ArxivDataset.to_exclude_json.exists():
        with open(config.ArxivDataset.to_exclude_json, 'w') as f:
            json.dump([], f)
    
    # 1. 如果还没有 filter list，先生成 (如果已有，这步会直接读取现有的)
    # dblp_res = _dblp_parse()
    
    # 2. 运行只转换不下载的流程
    _convert_only_pipeline()