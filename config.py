import json
import sys
import pprint
import datetime
import pandas as pd

from pathlib import Path
from os import makedirs, listdir
from os.path import join, exists
from typing import List
from abc import ABCMeta, abstractmethod
from collections import namedtuple


from utils import natural_keys, papername_to_number

ACCEPT = "Accept"
REJECT = "Reject"

DOC_TYPE = 'txt'  # ['abs', txt]
DOC2VEC_TYPE = 'PV-DM'  # ['PV-DM', 'PV-DBOW'] Default: DM
EXPERIMENT_TYPE = 'WORecentBig4'  # ['WORecentBig4', 'WORecentBig4', 'Both']

if exists('/home/ln932/CSP'):
    ENV = 'CSP'
    RAW_DIR_1 = Path('/home/ln932/GoogleDrive/Security Review Dataset')
    RAW_DIR_2 = Path('/home/ln932/GoogleDrive/DeepSecPaperReview')
    ROOT_DIR = Path('/home/ln932/projects/SecReview')
elif exists('/home/ln932/Dell-Ubuntu'):
    ENV = 'local'
    ROOT_DIR = Path("/home/ln932/workspace/SecReview/")
elif exists('/home/niu/Manjaro'):
    ENV = 'local'
    ROOT_DIR = Path("/home/niu/workspace/CSP/SecReview/")
# AutoDL Environment Support
elif exists('/root/SecReview'):
    ENV = 'autodl'
    ROOT_DIR = Path('/root/SecReview')
    # 将处理后的数据集保存到 autodl-tmp 目录，避免系统盘满
    CONF_DATA_ROOT = Path('/root/autodl-tmp/Data')
    RAW_DIR_1 = Path('/root/autodl-tmp/Data/Security Review Dataset/Security Review Dataset')
    RAW_DIR_2 = Path('/root/autodl-tmp/Data/Security Review Dataset/Security Review Dataset')
    # RAW_DIR_1 = ROOT_DIR.joinpath('Data/Security Review Dataset')
    # RAW_DIR_2 = ROOT_DIR.joinpath('Data/Security Review Dataset')
elif exists('G:/SecReview2') or exists(r'G:\SecReview2'):
    ENV = 'local_windows'
    print("✅ 检测到本地 Windows 环境: G:/SecReview2")

    # 1. 设定项目根目录
    ROOT_DIR = Path('G:/SecReview2')

    # 2. 设定 Arxiv 数据路径 (最关键的部分)
    # 根据您的图片，结构是: SecReview2 -> arxiv -> Arxiv_Kaggle_Data
    ARXIV_DATA_ROOT = ROOT_DIR.joinpath('arxiv/Arxiv_Kaggle_Data')

    # 3. 设定会议论文数据路径 (shuju)
    # ⚠️ 注意：图片里没看到 'shuju' 文件夹，但我假设它依然在根目录下
    # 如果您的 Security Review Dataset 在其他地方，请修改这里
    CONF_DATA_ROOT = ROOT_DIR.joinpath('shuju')
    RAW_DIR_1 = CONF_DATA_ROOT.joinpath('Security Review Dataset')
    RAW_DIR_2 = CONF_DATA_ROOT.joinpath('Security Review Dataset')
else:
    sys.exit("Config.py panic! Unseen environment!")

CODE_PATH = ROOT_DIR.joinpath("Code/DeepSecReview")
RESOURCE_PATH = CODE_PATH.joinpath("resourses")
DB_PATH = CODE_PATH.joinpath("databases")
LOG_PATH = CODE_PATH.joinpath("logs/2023Mar")

MODEL_PATH = ROOT_DIR.joinpath('Models')


ARXIV_JSON_PATH_OLD = ARXIV_DATA_ROOT.joinpath("cs_CR.json")
ARXIV_JSON_PATH = ARXIV_DATA_ROOT.joinpath("cs_CR_2023Mar13.json")  # version 120
ARXIV_PDFS_PATH = ARXIV_DATA_ROOT.joinpath('papers')
DBLP_RESULT_PATH = RESOURCE_PATH.joinpath("dblp_results_2026.json")


class DatasetConfig(metaclass=ABCMeta):

    def __init__(self):
        self.name = "NoNameDataset"
        self.txt_dir = ""

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def __len__(self):
        pass

    @abstractmethod
    def get_pdf_file_list(self):
        pass

    @abstractmethod
    def get_txt_file_list(self):
        pass

    @abstractmethod
    def get_abs_file_list(self):
        pass

    @abstractmethod
    def get_pp_txt_file_list(self):
        pass

    @abstractmethod
    def get_pp_abs_file_list(self):
        pass


# This is a class/type alias
DatasetList = List[DatasetConfig]


def len_of_DSL(dsl: DatasetList):
    '''return length of dataset list'''
    return sum(map(len, dsl))


def number_of_files_in_DSL(dsl: DatasetList):
    for ds in dsl:
        print(ds.name)
        lengths = (
            len(ds.get_pdf_file_list()),
            len(ds.get_pp_txt_file_list()),
            len(ds.get_pp_abs_file_list()),
        )
        print("# of PDF, # of PPTXT, # of PPABS")
        print("{:>8}, {:>10}, {:>10}".format(*lengths))


class ConfDatasetConfig(DatasetConfig):
    def __init__(self,
                 ROOT: str,
                 RAW_DIR: str,
                 PDF: str = "papers",
                 TXT: str = "txt",
                 TXT_PP: str = "txt_preprocessed",
                 ABS: str = "abstract",
                 ABS_PP: str = "abstract_preprocessed",
                 VEC: str = "doc_vectors.npy",
                 # FILTER:  list = [],
                 ):

        self.root_dir = join(CONF_DATA_ROOT, ROOT)
        self.txt_dir = join(self.root_dir, TXT)
        self.txt_pp_dir = join(self.root_dir, TXT_PP)  # preprocessed
        self.abs_dir = join(self.root_dir, ABS)  # abstract
        self.abs_pp_dir = join(self.root_dir, ABS_PP)  # preprocessed
        self.vec = join(self.root_dir, VEC)

        for directory in [self.root_dir, self.txt_dir, self.txt_pp_dir, self.abs_dir, self.abs_pp_dir]:
            if not exists(directory):
                makedirs(directory)

        self.raw_dir = RAW_DIR
        self.pdf_dir = join(self.raw_dir, PDF)
        # self.filter_list   = FILTER
        self.name = ROOT
        self.len = len(self.get_pdf_file_list())

    def __str__(self):
        return f"{self.name}:\n  ROOT:{self.root_dir}\n  PDF:{self.pdf_dir}\n  Length:{self.len}"  # \n  SCORE:{self.score}"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.len

    def get_pdf_file_list(self):
        if not exists(self.pdf_dir):
            return []
        pdf_file_list = sorted(listdir(self.pdf_dir), key=natural_keys)
        # print(len(pdf_file_list), pdf_file_list)
        pdf_file_list = [join(self.pdf_dir, x) for x in pdf_file_list if '.pdf' in x.lower()]
        return pdf_file_list

    def get_txt_file_list(self):
        txt_file_list = sorted(listdir(self.txt_dir), key=natural_keys)
        txt_file_list = [join(self.txt_dir, x) for x in txt_file_list]
        return txt_file_list

    def get_abs_file_list(self):
        abs_file_list = sorted(listdir(self.abs_dir), key=natural_keys)
        abs_file_list = [join(self.abs_dir, x) for x in abs_file_list]
        return abs_file_list

    def get_pp_txt_file_list(self):
        pp_txt_file_list = sorted(listdir(self.txt_pp_dir), key=natural_keys)
        # if self.filter_list:
        #     pp_txt_file_list = filter(lambda x: papername_to_number(x) in self.filter_list, pp_txt_file_list)
        # print(f"{len(pp_txt_file_list)} files.")
        pp_txt_file_list = [join(self.txt_pp_dir, x) for x in pp_txt_file_list]
        return pp_txt_file_list

    def get_pp_abs_file_list(self):
        pp_abs_file_list = sorted(listdir(self.abs_pp_dir), key=natural_keys)
        # if self.filter_list:
        #     pp_abs_file_list = filter(lambda x: papername_to_number(x) in self.filter_list, pp_abs_file_list)
        pp_abs_file_list = [join(self.abs_pp_dir, x) for x in pp_abs_file_list]
        return pp_abs_file_list


ConfDatasets = []
ConfSecDatasets = []
ConfBig4Datasets = []

ConfDatasets_WORecentBig4 = []
ConfSecDatasets_WORecentBig4 = []
ConfBig4Datasets_WORecentBig4 = []
RecentBig4Datasets = []
recent_big4_confs = [
    'USENIX22', 'USENIX23', 'USENIX24', 'USENIX25',
    'SP22', 'SP23', 'SP24', 
    'NDSS22', 'NDSS23', 'NDSS24', 'NDSS25',
    'CCS22', 'CCS23', 'CCS24','CCS25'
]

df: pd.DataFrame = pd.read_csv(join(RAW_DIR_1, 'meta.csv'))
for conf, path in df.itertuples(index=False, name=None):
    # print(conf, path)
    # For general experiment
    ConfDatasets.append(ConfDatasetConfig(
        RAW_DIR=RAW_DIR_1,
        PDF=path,
        ROOT=conf
    ))
    if path.startswith('security') or path.startswith('Cryptography'):
        ConfSecDatasets.append(ConfDatasetConfig(
            RAW_DIR=RAW_DIR_1,
            PDF=path,
            ROOT=conf
        ))
    if path.startswith('security') and path.split('/')[1] in ['CCS', 'NDSS', 'SP', 'USENIX']:
        ConfBig4Datasets.append(ConfDatasetConfig(
            RAW_DIR=RAW_DIR_1,
            PDF=path,
            ROOT=conf
        ))

    # For novelty experiment
    if conf in recent_big4_confs:
        RecentBig4Datasets.append(ConfDatasetConfig(
            RAW_DIR=RAW_DIR_1,
            PDF=path,
            ROOT=conf
        ))
    else:  # conf is not recent big4
        ConfDatasets_WORecentBig4.append(ConfDatasetConfig(
            RAW_DIR=RAW_DIR_1,
            PDF=path,
            ROOT=conf
        ))
        if path.startswith('security') or path.startswith('Cryptography'):
            ConfSecDatasets_WORecentBig4.append(ConfDatasetConfig(
                RAW_DIR=RAW_DIR_1,
                PDF=path,
                ROOT=conf
            ))
        if path.startswith('security') and path.split('/')[1] in ['CCS', 'NDSS', 'SP', 'USENIX']:
            ConfBig4Datasets_WORecentBig4.append(ConfDatasetConfig(
                RAW_DIR=RAW_DIR_1,
                PDF=path,
                ROOT=conf
            ))


class ArxivDatasetConfig(DatasetConfig):
    def __init__(self,
                 # ROOT:   str,
                 # RAW_DIR:str,
                 PDF: str = "papers",
                 TXT: str = "txt",
                 TXT_PP: str = "txt_preprocessed",
                 ABS: str = "abstract",
                 ABS_PP: str = "abstract_preprocessed",
                 VEC: str = "doc_vectors.npy",
                 # limit: 如果为 None，则使用 arxiv_filter.json 中的全部论文；
                 #        否则只使用前 limit 篇（保持与旧实验兼容）
                 limit: int = None
                 # SCORE:  str = "",
                 # TOPIC:  str = "",
                 # FILTER:  list = [],
                 ):

        self.root_dir = ARXIV_DATA_ROOT
        self.name = "Arxiv_Kaggle_Data"
        self.pdf_dir = self.root_dir.joinpath(PDF)
        self.txt_dir = self.root_dir.joinpath(TXT)
        self.txt_pp_dir = self.root_dir.joinpath(TXT_PP)  # preprocessed
        self.abs_dir = self.root_dir.joinpath(ABS)  # abstract
        self.abs_pp_dir = self.root_dir.joinpath(ABS_PP)  # preprocessed
        self.vec = self.root_dir.joinpath(VEC)
        self.to_exclude_json = self.root_dir.joinpath("to_exclude.json")
        self.filter_json = self.root_dir.joinpath("arxiv_filter.json")
        with open(self.filter_json, 'r') as f:
            all_ids = json.load(f)

        # =========================
        # 负样本数量控制策略
        # =========================
        # 目标：默认情况下，让 *负样本（Arxiv）数量* 与 *正样本（会议论文）数量* 尽量一致，
        #      这样在训练 / 分类时正负样本规模对称。
        #
        # 正样本数量：使用所有会议数据集 ConfDatasets 的总论文数作为基准
        from config import len_of_DSL, ConfBig4Datasets  # 延迟导入以避免类型检查器告警
        positive_total = len_of_DSL(ConfBig4Datasets)

        # 如果外部没有显式指定 limit，则按「正样本数量」截断 Arxiv ID 列表；
        # 如果指定的 limit 比正样本还大，也同样截断到正样本数量。
        if limit is None:
            effective_limit = positive_total
        else:
            effective_limit = 28

        # 遍历 all_ids，只保留有对应 txt 文件存在的论文 ID，
        # 直到达到目标数量 effective_limit
        self.filter_list = []
        skipped_count = 0
        for paper_id in all_ids:
            if len(self.filter_list) >= effective_limit:
                break
            # 构建 txt 文件路径并检查是否存在
            txt_filename = paper_id.replace('/', '_') + '.txt'
            txt_path = self.txt_dir.joinpath(txt_filename)
            if txt_path.exists():
                self.filter_list.append(paper_id)
            else:
                skipped_count += 1

        # 如果跳过了一些文件，输出提示信息
        if skipped_count > 0:
            print(f"[ArxivDatasetConfig] 跳过了 {skipped_count} 篇没有 txt 文件的论文")
        
        # 检查是否达到目标数量
        if len(self.filter_list) < effective_limit:
            print(f"[ArxivDatasetConfig] 警告：只找到 {len(self.filter_list)} 篇有效论文，"
                  f"少于目标数量 {effective_limit}")

        # 实际使用的负样本数量（供其他模块参考）
        self.len_limit = len(self.filter_list)
        self.len = len(self.filter_list)

        for directory in [self.root_dir, self.txt_dir, self.txt_pp_dir, self.abs_dir, self.abs_pp_dir]:
            if not exists(directory):
                makedirs(directory)

    def __str__(self):
        return f"Dataset {self.name}:\n  ROOT:{self.root_dir}\n  PDF:{self.pdf_dir}\n  Length:{self.len}"  # SCORE:{self.score}"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.len

    @staticmethod
    def slugify(paper_id: str, suffix='pdf'):
        # 检查实际文件是否存在v1后缀，如果不存在则不加v1
        base_name = paper_id.replace('/', '_')
        # 对于txt文件，实际文件名可能没有v1后缀
        if suffix == 'txt':
            return base_name + '.' + suffix
        return base_name + 'v1.' + suffix

    def get_pdf_file_list(self):
        pdf_file_list = sorted([join(self.pdf_dir, self.slugify(paper_id)) for paper_id in self.filter_list],
                               key=natural_keys)
        return pdf_file_list

    def get_txt_file_list(self):
        txt_file_list = sorted(
            [join(self.txt_dir, self.slugify(paper_id, suffix='txt')) for paper_id in self.filter_list],
            key=natural_keys)
        return txt_file_list

    def get_abs_file_list(self):
        abs_file_list = sorted(
            [join(self.abs_dir, self.slugify(paper_id, suffix='txt')) for paper_id in self.filter_list],
            key=natural_keys)
        return abs_file_list

    def get_pp_txt_file_list(self):
        pp_txt_file_list = sorted(
            [join(self.txt_pp_dir, self.slugify(paper_id, suffix='txt')) for paper_id in self.filter_list],
            key=natural_keys)
        return pp_txt_file_list

    def get_pp_abs_file_list(self):
        pp_abs_file_list = sorted(
            [join(self.abs_pp_dir, self.slugify(paper_id, suffix='txt')) for paper_id in self.filter_list],
            key=natural_keys)
        return pp_abs_file_list


ArxivDataset = ArxivDatasetConfig()

# =========================
# 数据集组合配置说明（用于不同实验）
# =========================
# - AllDatasets:      所有会议 + Arxiv（主要用于 Doc2Vec 训练）
# - SecDatasets:      安全相关会议 + Arxiv
# - ClfDatasets:      分类实验使用的数据集：
#                     现在修改为「所有会议论文都作为正样本」，
#                     即 ConfDatasets 中的全部会议论文 = 正样本，
#                     ArxivDataset = 负样本。
ArxivDataset = ArxivDatasetConfig()
AllDatasets = ConfDatasets + [ArxivDataset]
SecDatasets = ConfSecDatasets + [ArxivDataset]
ClfDatasets = ConfBig4Datasets + [ArxivDataset]

ArxivDataset_WORecentBig4 = ArxivDatasetConfig(limit=len_of_DSL(ConfBig4Datasets_WORecentBig4))
AllDatasets_WORecentBig4 = ConfDatasets_WORecentBig4 + [ArxivDataset_WORecentBig4]
SecDatasets_WORecentBig4 = ConfSecDatasets_WORecentBig4 + [ArxivDataset_WORecentBig4]
ClfDatasets_WORecentBig4 = ConfBig4Datasets_WORecentBig4 + [ArxivDataset_WORecentBig4]

# Doc2Vec configuration
D2V_CONFIG = {
    'dm': 0 if DOC2VEC_TYPE == 'PV-DBOW' else 1,
    'vector_size': 400,
    'window': 5,
    'min_count': 10,
    'workers': 22,
    'epochs': 200,
    'alpha': 0.025,
    "min_alpha": 0.0001,
}
# VECTOR_PATH = join(DATA_ROOT, 'all_vectors_' + str(D2V_CONFIG['vector_size']) + '.npy')
VECTOR_PATH = join(CONF_DATA_ROOT, f"{DOC_TYPE}_vectors_{DOC2VEC_TYPE}_{str(D2V_CONFIG['vector_size'])}.npy")
REPORT_CSV_PATH = join(CONF_DATA_ROOT, f"analysis_report_{DOC_TYPE}_{DOC2VEC_TYPE}.csv")

VECTOR_PATH_WORecentBig4 = join(CONF_DATA_ROOT,
                                f"WORecentBig4_{DOC_TYPE}_vectors_{DOC2VEC_TYPE}_{str(D2V_CONFIG['vector_size'])}.npy")
VECTOR_PATH_RecentBig4 = join(CONF_DATA_ROOT,
                              f"RecentBig4_{DOC_TYPE}_vectors_{DOC2VEC_TYPE}_{str(D2V_CONFIG['vector_size'])}.npy")


class ExpDatasets:
    def __init__(self, accept_ds: DatasetList, reject_ds: DatasetList):
        self.accept_ds = accept_ds
        self.reject_ds = reject_ds

    def __str__(self):
        return f"""
Accepted dataset(len={len_of_DSL(self.accept_ds)}):
  {pprint.pformat(self.accept_ds, indent=1)}
  
Rejected dataset(len={len_of_DSL(self.reject_ds)}):
  {pprint.pformat(self.reject_ds, indent=1)}
"""


class ExperimentConfig:

    def __init__(self,
                 name,
                 exp_datasets: ExpDatasets,
                 paper_type: str,
                 params=None,
                 db_file=None,
                 log_file=None):
        self.name = name
        self.paper_type = paper_type
        self.datasets = exp_datasets
        date_str = datetime.date.today().strftime("%b%d")
        self.db_file = DB_PATH.joinpath(f"{self.name}_{self.paper_type}_{date_str}.db") if db_file is None else db_file
        self.log_file = LOG_PATH.joinpath(f"{self.name}_{self.paper_type}_{date_str}_exp.log") if log_file is None else log_file
        self.params = {} if params is None else params


# Top 4 confs held after 2021 September
# _exp_gpt_pos_ds = [ds for ds in AllDatasets if ds.name in ["CCS21", "CCS22", "NDSS22", "NDSS23", "SP22", "USENIX22"]]
_exp_gpt_pos_ds = [ds for ds in AllDatasets if ds.name in ["NDSS25"]]
_exp_gpt_neg_ds = [ArxivDatasetConfig(limit=len_of_DSL(_exp_gpt_pos_ds))]
GPT_EXP_CFG_ABS = ExperimentConfig(name="ChatGPT11",
                                   exp_datasets=ExpDatasets(
                                       accept_ds=_exp_gpt_pos_ds,
                                       reject_ds=_exp_gpt_neg_ds,
                                   ),
                                   paper_type='abs',
                                   params={
                                       # "model": "gpt-3.5-turbo"
                                       "model": "deepseek-coder"
                                   },
                                   )
GPT_EXP_CFG_TXT = ExperimentConfig(name="ChatGPT11",
                                   exp_datasets=ExpDatasets(
                                       accept_ds=_exp_gpt_pos_ds,
                                       reject_ds=_exp_gpt_neg_ds,
                                   ),
                                   paper_type='txt',
                                   params={
                                       # "model": "gpt-3.5-turbo"
                                       "model": "deepseek-coder"
                                   },
                                   )

if __name__ == '__main__':
    # number_of_files_in_DSL(ConfBig4Datasets)
    print(GPT_EXP_CFG_ABS.datasets)
    number_of_files_in_DSL([ArxivDataset])

    print(f'''Number of papers in datasets:
len_of_DSL(ConfDatasets)    : {len_of_DSL(ConfDatasets)}
len_of_DSL(SecDatasets)     : {len_of_DSL(SecDatasets)}
len_of_DSL(ConfSecDatasets) : {len_of_DSL(ConfSecDatasets)}
len_of_DSL(ConfBig4Datasets): {len_of_DSL(ConfBig4Datasets)}
len(ArxivDataset)           : {len(ArxivDataset)}
''')

    # print("="*40)

    print(f'''Number of papers in datasets without recent years big 4:
len_of_DSL(RecentBig4Datasets)           : {len_of_DSL(RecentBig4Datasets)}
len_of_DSL(ConfDatasets_WORecentBig4)    : {len_of_DSL(ConfDatasets_WORecentBig4)}
len_of_DSL(SecDatasets_WORecentBig4)     : {len_of_DSL(SecDatasets_WORecentBig4)}
len_of_DSL(ConfSecDatasets_WORecentBig4) : {len_of_DSL(ConfSecDatasets_WORecentBig4)}
len_of_DSL(ConfBig4Datasets_WORecentBig4): {len_of_DSL(ConfBig4Datasets_WORecentBig4)}
len(ArxivDataset_WORecentBig4)           : {len(ArxivDataset_WORecentBig4)}
''')

    # for c in ConfDatasets:
    #     print(c.name, ':', len(c))
    # for c in ConfNewlyAddedDatasets:
    #     print(c)
    # print(len_of_DSL(ConfNewlyAddedDatasets)) # 1085
    # t = ArxivDataset.get_pp_txt_file_list()
    # p = ArxivDataset.get_pdf_file_list()
    # a = ArxivDataset.get_pp_abs_file_list()
