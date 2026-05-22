# Based on the post of https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72

import spacy
import pandas as pd
# import numpy as np
import nltk
from nltk.tokenize.toktok import ToktokTokenizer

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import unicodedata
import shlex, subprocess
import logging
import traceback
import os
import re

import config
from config import DatasetConfig, DatasetList
import utils

from resourses.contractions import CONTRACTION_MAP


try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    nlp = spacy.load('en', parse=True, tag=True, entity=True)
nlp.max_length = 2000000
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
publication_word_list = ['acm', 'ieee', 'proceeding', 'symposium', 'copyright', 'usenix', 'oakland', 'ndss', 'ccs', 'sp',
                         'eurocrypt', 'association', 'doi', 'grant']


# def strip_html_tags(text):
#     soup = BeautifulSoup(text, "html.parser")
#     stripped_text = soup.get_text()
#     return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
            if contraction_mapping.get(match)\
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ !=
                     '-PRON-' else word.text for word in text])
    return text


def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [
            token for token in tokens if token not in stopword_list]
        filtered_tokens = [
            token for token in tokens if token not in publication_word_list]
    else:
        filtered_tokens = [
            token for token in tokens if token.lower() not in stopword_list]
        filtered_tokens = [
            token for token in tokens if token.lower() not in publication_word_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_reference_number(text):
    pattern = r"\[\s*\d*\s*\]"
    text = re.sub(pattern, '', text)
    return text

def self_defined_processing(text, dataset_name: str):
    # 引入正则模块（虽然文件头导入了，但在函数里确保可用）
    import re

    def remove_publications(text):
        if dataset_name.startswith('SP'):
            # 尝试找到第一个双换行，跳过前面的出版信息
            # 如果找不到，就保留原样，防止报错
            idx = text.find('\n\n')
            if idx != -1:
                text = text[idx+2:]
        if dataset_name.startswith('Arxiv'):
            tl = text.split('\n')
            i = 0
            for line in tl:
                if len(line) <= 2:
                    i += 1
                else:
                    break
            text = '\n'.join(tl[i:])
        return text

    def remove_authors(text):
        # ===【修复核心】===
        # 原代码: text = text[:text.find('\n\n') + 1] + text[text.find('\nabstract'):]
        # 问题: SP 格式是 "Abstract—" (破折号)，find('\nabstract') 返回 -1，导致切片错误。
        
        # 1. 确定标题结束的大致位置（假设前 2000 字符内出现的第一个双换行是标题结束）
        # 找不到也不要紧，设为 0
        title_end_match = re.search(r'\n\n', text[:3000])
        title_end = title_end_match.start() + 1 if title_end_match else 0
        
        # 2. 智能寻找 Abstract 的开始位置
        # 兼容: 
        #   \nAbstract (标准)
        #   Abstract— (SP/IEEE)
        #   Abstract. (Crypto/Springer)
        #   A B S T R A C T (某些旧格式)
        abs_pattern = r'(?:^|\n)\s*(?:abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t)\s*[:\.\—\-\s]'
        abs_match = re.search(abs_pattern, text, re.IGNORECASE)
        
        if abs_match:
            # 找到了明确的 Abstract 标记
            # 保留标题部分 + Abstract 及其之后的内容
            # 注意：如果 title_end 计算不准，我们优先信任 abs_match 的位置
            start_idx = abs_match.start()
            # 如果标题结束位置比摘要开始位置还晚，那肯定找错了，重置标题结束位置
            if title_end > start_idx:
                title_end = 0
                
            return text[:title_end] + "\n" + text[start_idx:]
        else:
            # 没找到 Abstract，尝试找 Introduction 往前倒推
            # 避免像旧代码那样直接返回空文件
            intro_match = re.search(r'(?:^|\n)\s*(?:1\.?|i\.?)?\s*introduction', text, re.IGNORECASE)
            if intro_match:
                # 假设 Introduction 之前的是正文/摘要，切掉前面的作者信息（虽然不完美，但比空着强）
                intro_start = intro_match.start()
                if title_end > intro_start: title_end = 0
                return text[:title_end] + "\n" + text[intro_start:]
            
            # 实在没招了，返回原文，千万别返回空字符串
            return text

    def remove_references(text):
        # 增强版：不仅找 references，还要防止把正文里的 references 单词误判
        # 通常参考文献在文章最后，且独占一行
        ref_match = re.search(r'\n\s*references\s*\n', text, re.IGNORECASE)
        if ref_match:
             text = text[:ref_match.start()]
        else:
            # 备用方案：找 [1] 这种引用列表密集的区域（可选，这里先用简单的）
            idx = text.lower().rfind('\nreferences\n')
            if idx != -1:
                text = text[:idx]
        return text

    text = remove_publications(text)
    text = remove_authors(text)
    text = remove_references(text)

    return text

def normalize_corpus(corpus, dataset_name, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True,
                     self_defined = True,
                     text_lemmatization=True, special_char_removal=True,
                     reference_removal=True,
                     stopword_removal=True, remove_digits=True):

    normalized_corpus = []
    # normalize each document in the corpus
    for idx, doc in enumerate(corpus):
        # print("Preprocessing", idx+1, "/", len(corpus))
        lines = doc.split('\n')
        # 保留条件：这一行长度大于1（正常单词），或者是空行（保留段落结构）
        # strip() 去除首尾空格，防止 " a " 这种被误判
        cleaned_lines = [line for line in lines if len(line.strip()) > 1 or len(line.strip()) == 0]
        doc = '\n'.join(cleaned_lines)
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)

        # expand contractions
        if contraction_expansion:
            doc = expand_contractions(doc)

        # lowercase the text
        if text_lower_case:
            doc = doc.lower()

        if self_defined:
            doc = self_defined_processing(doc, dataset_name)

        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)

        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)

        # remove special characters and\or digits
        if reference_removal:
            doc = remove_reference_number(doc)

        if special_char_removal:
            # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r'([\[{.(-)!}\]])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)
            doc = doc.replace('[','')
            doc = doc.replace(']','')

        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)

        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)

        normalized_corpus.append(doc)

    return normalized_corpus

def normalize_file(args, DEBUG = False):
    src, dst, ds_name = args
    logger = logging.getLogger()
    try:
        with open(src, 'r') as f:
            txt = f.read()
            txt = normalize_corpus([txt], dataset_name=ds_name)[0]
            if DEBUG:
                print(txt)
            else:
                with open(dst, 'w') as f:
                    f.write(txt)
        return src
    except Exception as e:
        logger.error(f"An exception occurred while normalizing {src}: {repr(e)}")
        traceback.print_exc()
        return f'Exception :{repr(e)}'

def generate_normalize_list(datasetlist: DatasetList, doc_type='txt'):
    norm_list = []
    assert doc_type in ['txt', 'abs']
    for dataset in datasetlist:
        ds_name = dataset.name
        if doc_type == 'txt':
            src_txt_path, dst_txt_path = dataset.txt_dir, dataset.txt_pp_dir
        elif doc_type == 'abs':
            src_txt_path, dst_txt_path = dataset.abs_dir, dataset.abs_pp_dir
        file_list = sorted(os.listdir(src_txt_path), key=utils.natural_keys)
        file_list = [x for x in file_list if '.txt' in x.lower()]
        src_list = [os.path.join(src_txt_path, f) for f in file_list]
        dst_list = [os.path.join(dst_txt_path, f) for f in file_list]
        ds_name_list = [ds_name] * len(src_list)
        norm_list.extend(zip(src_list, dst_list, ds_name_list))
    return norm_list

def run_normalize_parallel(file_list):
    if len(file_list) >= 16:
        n_worker = 16
    else:
        n_worker = 1
    # with Pool(n_worker) as pool:
    #     r = list(tqdm(pool.imap(normalize_file, file_list), total=len(file_list)))
    # with ThreadPoolExecutor(max_workers=n_worker) as executor:
    #     r = list(tqdm(executor.map(normalize_file, file_list), total=len(file_list)))
    with ProcessPoolExecutor(max_workers=n_worker) as executor:
        r = list(tqdm(executor.map(normalize_file, file_list), total=len(file_list)))
    return len(r)


def normalize_pipeline(dataset: DatasetConfig, counter, tqdm_bar, doc_type='txt', DEBUG=True):
    logger = logging.getLogger()
    assert doc_type in ['txt', 'abs']
    if doc_type == 'txt':
        src_txt_path, dst_txt_path = dataset.txt_dir, dataset.txt_pp_dir
    elif doc_type == 'abs':
        src_txt_path, dst_txt_path = dataset.abs_dir, dataset.abs_pp_dir
    file_list = sorted(os.listdir(src_txt_path), key=utils.natural_keys)
    file_list = [x for x in file_list if '.txt' in x.lower()]
    # print(f"Normalize: {dataset.name}")
    for idx, file_name in enumerate(file_list):
        # print(f"  {idx+counter}/11445: {file_name}", flush=True)
        # logger.info(f"  {idx+counter}/{config_arxiv.LEN_ALL_PDF}: {file_name}") #, flush=True)
        logger.info(f"  {idx+counter}: {file_name}") #, flush=True)
        with open(os.path.join(src_txt_path, file_name), 'r') as f:
            txt = f.read()
            txt = normalize_corpus([txt], dataset_name=dataset.name)[0]
            if DEBUG:
                print(txt)
                break
            else:
                with open(os.path.join(dst_txt_path, file_name), 'w') as f:
                    f.write(txt)
        tqdm_bar.update(1)
    return len(file_list)

def extract_abstract(txt):
    import re
    # 1. 找开头
    # 匹配 "Abstract", "A b s t r a c t", 后面跟点、冒号、破折号或空格
    start_pattern = r'(?:^|\s)(abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t)\s*[:\.\—\-\s]'
    match_start = re.search(start_pattern, txt, re.IGNORECASE)
    
    if not match_start:
        # 找不到摘要头，尝试直接返回前 1500 字（通常包含摘要）
        return txt[:1500]
        
    start_idx = match_start.start()

    # 2. 找结尾 (增强版，加入 ACM 的特征)
    # 我们只在找到的 abstract 之后开始搜索结尾
    search_txt = txt[start_idx:]
    
    end_patterns = [
        r'(?:1\.?|i\.?|1\s|i\s)?\s*introduction',  # 标准 Introduction
        r'index\s*terms',       # IEEE 常用
        r'key\s*words',         # 通用
        r'ccs\s*concepts',      # ACM (CCS, SIGCOMM) !!! 关键修正 !!!
        r'categories\s*and\s*subject',    # ACM 老版本
        r'general\s*terms'      # ACM 老版本
    ]
    
    # 合并正则
    combined_pattern = "|".join(end_patterns)
    match_end = re.search(combined_pattern, search_txt, re.IGNORECASE)
    
    if match_end:
        end_idx = start_idx + match_end.start()
    else:
        # 如果找不到这些结尾词，就截取一段合理的长度（比如 2000 字符）
        # 或者找第一个双换行（段落结束）
        end_idx = start_idx + 2000
    
    raw_abstract = txt[start_idx:end_idx].strip()
    
    # 3. 长度兜底
    # 如果提取出来的太短（比如提取到了空行），可能出错了，返回前段文本
    if len(raw_abstract) < 20:
        return txt[:1500] 
        
    return raw_abstract
def extract_abstract_pipeline(dataset: DatasetConfig, tqdm_bar, counter=0, DEBUG=True):
    logger = logging.getLogger()
    src_txt_path, dst_txt_path = dataset.txt_dir, dataset.abs_dir
    file_list = sorted(os.listdir(src_txt_path), key=utils.natural_keys)
    file_list = [x for x in file_list if '.txt' in x.lower()]
    print(f"Extracting Abstract for: {dataset.name}")
    for idx, file_name in enumerate(file_list):
        logger.info(f"  {idx+counter}/{len(dataset)}: {file_name}") #, flush=True)
        with open(os.path.join(src_txt_path, file_name), 'r') as f:
            txt = f.read()
            txt = extract_abstract(txt)
            if DEBUG:
                print(txt)
            else:
                with open(os.path.join(dst_txt_path, file_name), 'w') as f:
                    f.write(txt)
        tqdm_bar.update(1)
    return len(file_list)


def pdf2txt_dataset(dataset: DatasetConfig, tqdm_bar, counter = 0, DEBUG=True):
    logger = logging.getLogger()
    logger.info(f"PDF2TXT: {dataset.name}")
    file_list = dataset.get_pdf_file_list()
    for idx, pdf_file in enumerate(file_list):
        src = pdf_file
        dst = os.path.join(dataset.txt_dir, os.path.splitext(os.path.basename(pdf_file))[0] + '.txt')
        cmd = f'pdf2txt.py "{src}" -o "{dst}"'
        logger.info(f"{idx+counter}/{len(dataset)}: {cmd}")
        if not DEBUG:
            # ret = os.system(cmd)
            ret = subprocess.run(shlex.split(cmd), stderr=subprocess.PIPE)
            if ret.returncode != 0:
                # print(f"PDF2TXT ERROR: ret={ret}; {src}.")
                logger.error(f"{cmd} returned {ret.returncode}; info:\n{ret.stderr}.")
        tqdm_bar.update(1)
    return len(file_list)

def pdf2txt_parellel(dataset: DatasetConfig, n_thread = 20, DEBUG = True):
    logger = logging.getLogger()
    logger.info(f"PDF2TXT: {dataset.name}")
    file_list = dataset.get_pdf_file_list()
    def pdf2txt_helper(params_tuple):
        idx, pdf_file = params_tuple
        src = pdf_file
        dst = os.path.join(dataset.txt_dir, os.path.splitext(os.path.basename(pdf_file))[0] + '.txt')
        cmd = f'pdf2txt.py "{src}" -o "{dst}"'
        logger.info(f"{idx+1}/{len(dataset)}: {cmd}")
        if not DEBUG:
            ret = subprocess.run(shlex.split(cmd), stderr=subprocess.PIPE)
            if ret.returncode != 0:
                logger.error(f"{cmd} returned {ret.returncode}; info:\n{ret.stderr}.")
                return pdf_file
    # with Pool(n_thread) as pool:
    #     results = list(tqdm(pool.imap(pdf2txt_helper, enumerate(file_list)), total=len(file_list)))
    with ThreadPoolExecutor(max_workers=n_thread) as executor:
        results = list(tqdm(executor.map(pdf2txt_helper, enumerate(file_list)), total=len(file_list)))
    return results

def filter_data(dataset: DatasetConfig):
    df = pd.read_csv(dataset.score, encoding="latin1")
    df = df[['paper', 'decision']]
    df = df.drop_duplicates()
    print("#Entry before filtering:", len(df))
    df = df.dropna(how='any') # 472
    print("#Entry after filtering:", len(df))
    result = df.paper.to_list()
    print(dataset.root_dir, "paper list after filtering (Please add to config.py before next steps):")
    print(result)
    return result

def remove_first_page_usenix(usenix_confs):
    ''' just a reminder of how to remove the first page, actually done by manually input the cmd '''
    cmd_remove_1st_page = 'for i in *pdf ; do pdftk "$i" cat 2-end output "../$i" ; done'
    ConfUsenixSec = [d for d in config.ConfDatasets if d.name in usenix_confs]
    for d in ConfUsenixSec:
        cmd = f'cd "{d.pdf_dir}" ; mkdir untrimmed ; mv *pdf untrimmed ; cd untrimmed; {cmd_remove_1st_page}'
        print(cmd)


if __name__ == "__main__":
    FLAG_DEBUG = False

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger()
    # file_handler = logging.FileHandler('./logs/2023_Mar/preprocess_CCS22NDSS23.log')
    file_handler = logging.FileHandler(config.LOG_PATH.joinpath('preprocess_arxiv.log'))
    file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)s %(message)s'))
    file_handler.setLevel(logging.NOTSET)
    logger.addHandler(file_handler)

    # last_year_big4 = ['USENIX22', 'SP22', 'NDSS22', 'CCS21']
    # DS_todo = [d for d in config.AllDatasets if d.name in last_year_big4]
    # just a reminder of how to remove the first page, actually done by manually input the cmd
    # remove_first_page_usenix(usenix_confs=['USENIX22',])
    # DS_todo = [d for d in config.AllDatasets if d.name in ['NDSS23', 'CCS22', ]]
    # for d in DS_todo:
    #     pdf2txt_parellel(dataset=d, DEBUG=FLAG_DEBUG)
    # with tqdm() as tqdm_bar:
    #     for ds in DS_todo:
    #         extract_abstract_pipeline(ds, counter=0, tqdm_bar=tqdm_bar, DEBUG=FLAG_DEBUG)
    # norm_list = generate_normalize_list(DS_todo, doc_type='txt')
    # ret = run_normalize_parallel(norm_list)
    # norm_list = generate_normalize_list(DS_todo, doc_type='abs')
    # ret = run_normalize_parallel(norm_list)

    # DS_newly_added = [d for d in config.DS_Conf_Sec if d.name in ("USENIX16", "USENIX17", "USENIX18", "NDSS18", "NDSS19", "NDSS20", "NDSS21")]
    # DS_newly_added = config.ConfNewlyAddedDatasets
    # for d in DS_newly_added:
    #     pdf2txt_parellel(dataset=d, DEBUG=FLAG_DEBUG)
    # with tqdm() as tqdm_bar:
    #     for ds in DS_newly_added:
    #         extract_abstract_pipeline(ds, counter=0, tqdm_bar=tqdm_bar, DEBUG=FLAG_DEBUG)
    # norm_list = generate_normalize_list(DS_newly_added, doc_type='txt')
    # ret = run_normalize_parallel(norm_list)
    # norm_list = generate_normalize_list(DS_newly_added, doc_type='abs')
    # ret = run_normalize_parallel(norm_list)

    from config import ArxivDataset
    # print("PDF2TXT for Arxiv.")
    # logger.info("PDF2TXT for Arxiv.")
    # pdf2txt_parellel(dataset=ArxivDataset, DEBUG=FLAG_DEBUG) # already done in `arxiv_fetch.py`
    # print("Extract Abstract for Arxiv.")
    logger.info("\n\nExtract Abstract for Arxiv.")
    with tqdm(total=len(ArxivDataset)) as tqdm_bar:
        extract_abstract_pipeline(ArxivDataset, counter=0, tqdm_bar=tqdm_bar, DEBUG=FLAG_DEBUG)
    # print("Normalize TXT for Arxiv.")
    # logger.info("\n\nNormalize TXT for Arxiv.")
    # norm_list = generate_normalize_list([ArxivDataset], doc_type='txt')
    # ret = run_normalize_parallel(norm_list)
    # print("Normalize Abstract for Arxiv.")
    # logger.info("\n\nNormalize Abstract for Arxiv.")
    # norm_list = generate_normalize_list([ArxivDataset], doc_type='abs')
    # ret = run_normalize_parallel(norm_list)

    # i = 1
    # print("Extract Abstract:", flush=True)
    # with tqdm(total=config.LEN_ALL_PDF) as tqdm_bar:
    #     for dataset in config.SUBSET_1:
    #         i += extract_abstract_pipeline(dataset, counter=i, tqdm_bar=tqdm_bar, DEBUG=FLAG_DEBUG)
    #     for dataset in config.SUBSET_2:
    #         i += extract_abstract_pipeline(dataset, counter=i, tqdm_bar=tqdm_bar, DEBUG=FLAG_DEBUG)
    # print(f"All {i-1} abstracts are extracted.")

    # norm_list = generate_normalize_list(config.ALL_DOC_SET, doc_type='abs')
    # ret = run_normalize_parallel(norm_list)

    # i = 1
    # print("Normalize Abstract:", flush=True)
    # with tqdm(total=config.LEN_ALL_PDF) as tqdm_bar:
    #     for dataset in config.SUBSET_1:
    #         i += normalize_pipeline(dataset, counter=i, tqdm_bar=tqdm_bar, doc_type='abs', DEBUG=False)
    #     for dataset in config.SUBSET_2:
    #         i += normalize_pipeline(dataset, counter=i, tqdm_bar=tqdm_bar, doc_type='abs', DEBUG=False)
    # print(f"All {i} documents are normalized.")

    # # According to scores.csv files to filter
    # i = 1
    # print("PDF2TXT:", flush=True)
    # with tqdm(total=config.LEN_ALL_PDF) as tqdm_bar:
    #     for dataset in config.SUBSET_1:
    #         i += pdf2txt(dataset, counter=i, tqdm_bar=tqdm_bar, DEBUG=FLAG_DEBUG)
    #         # i += normalize_pipeline(dataset, counter=i, tqdm_bar=tqdm_bar, DEBUG=False)
    #     for dataset in config.SUBSET_2:
    #         i += pdf2txt(dataset, counter=i, tqdm_bar=tqdm_bar, DEBUG=FLAG_DEBUG)
    #         # i += normalize_pipeline(dataset, counter=i, tqdm_bar=tqdm_bar, DEBUG=False)
    # print(f"All {i} pdfs are turned into txt.")

    #
    # i = 1
    # for dataset in config.SUBSET_1:
    #     # i = pdf2txt(dataset.PDF, dataset.TXT, counter=i, DEBUG=False)
    #     i += normalize_pipeline(dataset, counter=i, DEBUG=False)
    # for dataset in config.SUBSET_2:
    #     # i = pdf2txt(dataset.PDF, dataset.TXT, counter=i, DEBUG=True)
    #     i += normalize_pipeline(dataset, counter=i, DEBUG=False)
    # print(f"All {i} documents are normalized.")

    # normalize_filelist([
    #     (
    #         "/home/ln932/projects/SecReview/Data/CVPR14/txt/Li_Persistence-based_Structural_Recognition_2014_CVPR_paper.txt",
    #         "/home/ln932/projects/SecReview/Data/CVPR14/txt_preprocessed/Li_Persistence-based_Structural_Recognition_2014_CVPR_paper.txt"
    #     ),
    #     (
    #         "/home/ln932/projects/SecReview/Data/CVPR14/txt/Xing_Towards_Multi-view_and_2014_CVPR_paper.txt",
    #         "/home/ln932/projects/SecReview/Data/CVPR14/txt_preprocessed/Xing_Towards_Multi-view_and_2014_CVPR_paper.txt"
    #     ),
    #     (
    #         "/home/ln932/projects/SecReview/Data/ICCV15/txt/Liu_Projection_Bank_From_ICCV_2015_paper.txt",
    #         "/home/ln932/projects/SecReview/Data/ICCV15/txt_preprocessed/Liu_Projection_Bank_From_ICCV_2015_paper.txt"
    #     ),
    #     (
    #         "/home/ln932/projects/SecReview/Data/ICCV15/txt/liu_Illumination_Robust_Color_ICCV_2015_paper.txt",
    #         "/home/ln932/projects/SecReview/Data/ICCV15/txt_preprocessed/liu_Illumination_Robust_Color_ICCV_2015_paper.txt"
    #     ),
    #     (
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt/P18-1037.txt",
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt_preprocessed/P18-1037.txt"
    #     ),
    #     (
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt/P18-1043.txt",
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt_preprocessed/P18-1043.txt"
    #     ),
    #     (
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt/P18-1046.txt",
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt_preprocessed/P18-1046.txt"
    #     ),
    #     (
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt/P18-1060.txt",
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt_preprocessed/P18-1060.txt"
    #     ),
    #     (
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt/P18-1077.txt",
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt_preprocessed/P18-1077.txt"
    #     ),
    #     (
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt/P18-1085.txt",
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt_preprocessed/P18-1085.txt"
    #     ),
    #     (
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt/P18-1086.txt",
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt_preprocessed/P18-1086.txt"
    #     ),
    #     (
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt/P18-1098.txt",
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt_preprocessed/P18-1098.txt"
    #     ),
    #     (
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt/P18-1125.txt",
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt_preprocessed/P18-1125.txt"
    #     ),
    #     (
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt/P18-1126.txt",
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt_preprocessed/P18-1126.txt"
    #     ),
    #     (
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt/P18-1199.txt",
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt_preprocessed/P18-1199.txt"
    #     ),
    #     (
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt/P18-1213.txt",
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt_preprocessed/P18-1213.txt"
    #     ),
    #     (
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt/P18-1229.txt",
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt_preprocessed/P18-1229.txt"
    #     ),
    #     (
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt/P18-1249.txt",
    #         "/home/ln932/projects/SecReview/Data/ACL18/txt_preprocessed/P18-1249.txt"
    #     ),
    # ], DEBUG=False)

    # norm_list = generate_normalize_list(config.ALL_DOC_SET)
    # ret = run_normalize_parallel(norm_list)

    # normalize_pipeline(config.USENIX18.TXT, config.USENIX18.TXT_PP, DEBUG=False)
    # normalize_pipeline(config.USENIX17, DEBUG=True)
    # normalize_pipeline(config.NDSS18, DEBUG=True)

    # filter_data(config.USENIX17)
    # filter_data(config.USENIX18)
    # filter_data(config.NDSS18)

    # for dataset in config.SUBSET_2:
    #     filter_data(dataset)
