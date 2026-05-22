import os
import sys
from typing import Callable, List
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import tiktoken
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from tqdm import tqdm

from config import ACCEPT, REJECT
from config import ExperimentConfig
from config import GPT_EXP_CFG_TXT, GPT_EXP_CFG_ABS
from db_util import DBManager, DataEntry

# 建议使用环境变量，或者在这里填入您的 Key
os.environ[
    "OPENAI_API_KEY"] = "sk-cd4bedc6f6204f218123b30e8d8481db"

openai.api_base = "https://api.deepseek.com"
# openai.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
exp_cfg = GPT_EXP_CFG_ABS
# exp_cfg = GPT_EXP_CFG_TXT

# 强制使用 gpt-3.5 或 gpt-4 的编码器，因为 tiktoken 不识别 deepseek 的名字
try:
    encoding = tiktoken.encoding_for_model(exp_cfg.params['model'])
except KeyError:
    encoding = tiktoken.get_encoding("cl100k_base")  # DeepSeek 兼容这个编码


def count_tokens(text):
    return len(encoding.encode(text))


# 【修改点1】将默认 max_tokens 极大提升，防止切分
def split_into_chunks(text, prompts=None, max_tokens=100000):
    # Deduct prompt length from `max_tokens`
    prompts_token_length = [0, 0]
    if prompts:
        # 如果是单次发送，这里其实用不到了，但为了兼容性保留逻辑
        if len(prompts) == 2:
            for prompt in prompts:
                prompt = prompt["content"].split()
                for p_word in prompt:
                    prompts_token_length[-1] += count_tokens(p_word)
        else:
            # 对于单条 prompt 的情况
            prompts_token_length[0] = 100  # 给 prompt 预留一点空间

    # Split by chunks
    words = text.split()
    chunks = []

    current_chunk = []
    current_chunk_tokens = 0

    token_limit = max_tokens - prompts_token_length[0]

    # 简单的分词统计
    for word in words:
        token_length = count_tokens(word)
        # 如果单个词就超长（极少见），强行截断或放入
        if current_chunk_tokens + token_length > token_limit:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_chunk_tokens = 0
            # 后续块的限制（如果有的话）
            token_limit = max_tokens - prompts_token_length[-1]

        current_chunk.append(word)
        current_chunk_tokens += token_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


class QueryHandler:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self._end_flag = "<|end_of_paper|>"
        self.arsenal = {
            'abs': [{
                "role": "system",
                "content": "You are an experienced and fair reviewer from top cybersecurity conferences (NDSS, IEEE S&P, CCS and USENIX Security). I will give you a paper abstract for you to read and review. I want you to decide whether this paper should be accepted or not. You must first tell me your decision with Accept or Reject, and then explain your reasons in detail."
            }],
            # 【修改点2】重写 txt_start，移除切块描述，直接要求评审全文
            'txt_start': [{
                "role": "system",
                "content": "You are an experienced and fair reviewer from top cybersecurity conferences (NDSS, IEEE S&P, CCS and USENIX Security). I will give you a complete research paper to read and review. Please read the full text carefully. I want you to decide whether this paper should be accepted or not. You must first tell me your decision with 'Accept' or 'Reject', and then explain your reasons in concise language. Focus on the contribution, methodology, and evaluation."
            }],
            # 下面这两个在 One-Shot 模式下其实用不到了
            'txt_chunk': [{
                "role": "system",
                "content": f'''Here comes a following chunk of the paper. Please only reply with "OK" if the text does not contain "{self._end_flag}".'''
            }],
            'txt_end': [{
                "role": "system",
                "content": '''Now you have read all the chunks from the paper, you can start to review now...'''
            }],
        }

    def create_message(self, paper_text):
        if self.cfg.paper_type == 'abs':
            return self.arsenal['abs'] + [{"role": "user", "content": paper_text}]

        elif self.cfg.paper_type == 'txt':
            # 【修改点3】TXT 模式不再走切块循环，直接构建单次请求
            # 1. 尝试不切分，直接看做一个大块
            chunks = split_into_chunks(paper_text, max_tokens=30000)

            # 如果真的太长（超过10万token），还是得切，但这种情况极少
            # 如果只有一个块（绝大多数情况），直接构造消息
            if len(chunks) == 1:
                messages = []
                # 构造唯一的对话历史：[System Prompt, User(全文)]
                single_turn = self.arsenal['txt_start'] + [{"role": "user", "content": chunks[0]}]
                messages.append(single_turn)
                return messages
            else:
                # 极其罕见的超长论文（>10万Token），回退到切分逻辑，或者直接报错
                # 这里我们简单处理：只取第一个块（通常包含了摘要、介绍和大部分内容），防止报错
                # 或者您可以选择只取前6万字符
                logger.warning("Paper too long (>100k tokens), truncation applied.")
                single_turn = self.arsenal['txt_start'] + [{"role": "user", "content": chunks[0]}]
                return [single_turn]

        else:
            assert self.cfg.paper_type in ('abs', 'txt')

    # https://platform.openai.com/docs/guides/rate-limits/overview
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def query(self, data_entry: DataEntry):
        assert self.cfg.paper_type == data_entry.paper_type

        paper_text = data_entry.paper_text
        # 这里返回的 messages 是一个列表的列表：[ [system, user_full_text] ]
        messages_list = self.create_message(paper_text)

        data_entry.messages = messages_list  # 保存一下用于记录

        outputs = []
        params = []

        if self.cfg.paper_type == 'abs':
            # abs 逻辑保持不变
            msg = messages_list  # abs 返回的是单层列表
            response = openai.ChatCompletion.create(
                model=exp_cfg.params['model'],
                messages=msg,
                # enable_search=False
            )
            result = ''
            for choice in response.choices:
                result += choice.message.content

            outputs = [result]
            params = [{'model': exp_cfg.params['model'], 'messages': msg}]

        elif self.cfg.paper_type == 'txt':
            # txt 逻辑：遍历 messages_list (现在通常长度为1)
            # logger.opt(raw=True).info(f"{len(messages_list)} chunks:") # 不需要打印 chunk 数了

            for i, msg in enumerate(messages_list):
                # logger.opt(raw=True).info(f" Sending full paper...")
                response = openai.ChatCompletion.create(
                    model=exp_cfg.params['model'],
                    messages=msg,
                    # enable_search=False
                )
                result = ''
                for choice in response.choices:
                    result += choice.message.content
                outputs.append(result)
                params.append({
                    'model': exp_cfg.params['model'],
                    'messages': msg
                })
                # logger.opt(raw=True).info(f" Done.")

        else:
            raise NotImplementedError

        data_entry.output = outputs
        data_entry.params = params
        return data_entry


class GPTDataLoader:
    def __init__(self, exp_cfg: ExperimentConfig):
        self.cfg = exp_cfg
        self.datasets = exp_cfg.datasets
        self.paper_type = exp_cfg.paper_type
        self.data_list = []
        i = 1
        for dsl, l in ((self.datasets.accept_ds, ACCEPT), (self.datasets.reject_ds, REJECT)):
            for ds in dsl:
                file_list = ds.get_abs_file_list() if self.paper_type == 'abs' else ds.get_txt_file_list()
                for paper_file in file_list:
                    with open(paper_file, 'r', encoding='utf-8', errors='ignore') as f:
                        paper_content = f.read()
                    self.data_list.append(DataEntry(id=i,
                                                    file=paper_file,
                                                    conf=ds.name,
                                                    label=l,
                                                    paper_text=paper_content,
                                                    paper_type=self.paper_type))
                    i += 1
        # assert i == 1988 + 1
        assert len([1 for d in self.data_list if d.label == ACCEPT]) == len(
            [0 for d in self.data_list if d.label == REJECT])


def _run_queries(errors_only=False):
    # Initializations
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(exp_cfg.log_file, level="TRACE")

    # Load Data
    data_loader = GPTDataLoader(exp_cfg)
    data_list = data_loader.data_list
    db = DBManager(db_file=exp_cfg.db_file)
    querier = QueryHandler(cfg=exp_cfg)

    # Filter errors
    if errors_only:
        import sqlite3

        # 1. 直接问数据库：谁出错了？
        print(f"🔧 [手动模式] 正在扫描数据库: {exp_cfg.db_file}")
        conn = sqlite3.connect(exp_cfg.db_file)
        c = conn.cursor()

        # 选出 pred 是 'Error' 或者还没有结果 (NULL) 的 ID
        c.execute("SELECT id FROM response WHERE pred = 'Error' OR pred IS NULL")
        error_ids = set([row[0] for row in c.fetchall()])
        conn.close()

        print(f"🧐 发现 {len(error_ids)} 个任务需要重试...")

        # 2. 从任务列表中只保留这些 ID
        # 只要 ID 对得上就跑，不再管顺序和总数
        data_list = [d for d in data_list if d.id in error_ids]

        if not data_list:
            print("🎉 所有任务都已完成，没有发现错误！")
        else:
            logger.info(f"Retrying {len(data_list)} failed tasks.")
        # err_indices = db.get_err_indices(rearrange=True)
        # # 注意：这里假设 id 是连续的且从1开始，如果出错可能需要更严谨的过滤
        # valid_indices = [i for i in err_indices if i < len(data_list)]
        # data_list = [data_list[i] for i in valid_indices]
        if data_list:
            logger.info(f"Retrying {len(data_list)} failed tasks.")

    logger.info(f"Query ({'errors_only' if errors_only else 'all'}) start! Total tasks: {len(data_list)}")

    # 定义任务：只负责预测，不负责写库
    def process_item_task(item):
        try:
            # 网络请求 (耗时)
            processed_item = querier.query(item)
            # 推理结果 (CPU)
            processed_item.infer_pred()
            return processed_item, None
        except Exception as e:
            item.output = [str(e)]
            item.pred = "Error"
            return item, e

    # 配置线程池
    MAX_WORKERS = 1

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 1. 提交所有任务
        future_to_item = {executor.submit(process_item_task, item): item for item in data_list}

        # 2. 在主线程中获取结果并写入数据库 (解决 SQLite 线程报错的核心)
        for future in tqdm(as_completed(future_to_item), total=len(data_list), desc="Processing"):
            item, error = future.result()

            # 【写入数据库】这一步现在回到了主线程执行，非常安全
            if errors_only:
                db.update(item)
            else:
                db.insert(item)

            if error:
                logger.error(f"Paper {item.id}: {item.file}. Error: {error}")
            else:
                logger.debug(f"Paper {item.id} Finished. Pred: {item.pred}")

    db.close()
    logger.info("All queries finished.")


# ... (省略了 _debug, _test_chunks, _first_abs_exp_results, GPTExpAnalysis 类, 保持原样即可) ...
def _debug():
    data_list = GPTDataLoader(exp_cfg).data_list
    querier = QueryHandler(cfg=exp_cfg)

    # Query loop
    logger.info("Debug start!")
    for i in tqdm(range(len(data_list[:1]))):
        # try:
        data_list[i] = querier.query(data_list[i])
        # data_list[i].pred = data_list[i].output.split('\n', 1)[0]  # TODO: need to improve
        # logger.info(f"Paper({i + 1}/{len(data_list)}): {data_list[i].file}. Response:\n{data_list[i].output}.")
        # except Exception as e:
        #     logger.error(f"Paper({i + 1}/{len(data_list)}): {data_list[i].file}. Error:\n{e}.")
        # sleep(10) # No need because: https://platform.openai.com/docs/guides/rate-limits/overview
        # input("press enter to continue")


def _test_chunks():
    data_list = GPTDataLoader(exp_cfg).data_list
    text = data_list[0].paper_text
    chunks = split_into_chunks(text, max_tokens=3900)
    for i in range(len(chunks)):
        with open(f"temp/chunks/chunks_{i + 1}.txt", 'w') as f:
            f.write(chunks[i])
    return chunks


def _first_abs_exp_results():
    print("NoAns: 101/1963=", 101 / 1963)
    print("Number of Accept: 1416/1963=", 1416 / 1963)
    print("Number of Reject: 446/1963=", 446 / 1963)
    print()
    print("Accuracy: 1048/1963=", 1048 / 1963)
    print("Accuracy for big4  (positive samples): 786/1963=", 786 / 983)
    print("Accuracy for arxiv (negative samples): 262/983=", 262 / 980)


# TODO
class GPTExpAnalysis:
    def __init__(self, exp_cfg: ExperimentConfig, db_file):
        self.cfg = exp_cfg
        self.db_file = db_file
        self.db = DBManager(db_file=db_file)
        self.data_list: List[DataEntry] = []

    def update_pred(self):
        pass
        # results = self.db.load()
        # TODO
        # self.db.update_pred()
        # pass

    def num_of(self, func: Callable):
        return len(list(filter(func, self.data_list)))

    def run(self):
        self.data_list = self.db.load_all()
        n = len(self.data_list)
        if self.cfg.paper_type == 'txt':
            assert n == 1988

        for i in range(len(self.data_list)):
            # self.data_list[i].infer_pred()
            pass

        n_err = self.num_of(lambda d: d.pred == "Error")
        n_ans = self.num_of(lambda d: d.pred in (ACCEPT, REJECT))
        n_ac = self.num_of(lambda d: d.pred == ACCEPT)
        n_rj = self.num_of(lambda d: d.pred == REJECT)
        n_ans_pos = self.num_of(lambda d: d.conf != "Arxiv_Kaggle_Data" and d.pred in (ACCEPT, REJECT))
        n_ans_neg = self.num_of(lambda d: d.conf == "Arxiv_Kaggle_Data" and d.pred in (ACCEPT, REJECT))
        n_correct_pos = self.num_of(lambda d: d.conf != "Arxiv_Kaggle_Data" and d.pred == ACCEPT)
        n_correct_neg = self.num_of(lambda d: d.conf == "Arxiv_Kaggle_Data" and d.pred == REJECT)
        n_correct = n_correct_pos + n_correct_neg
        # n_correct = self.num_of(lambda d: d.pred == d.label) # TODO: for some reason, this gives out wrong answer

        # Number of NoAns(Error + NoPred): {n - n_ans}
        # Number of Errors: {n_err}
        # Number of Records in DB: {n}
        print(f'''Analysis of {self.db_file}:
Number of Queries: 1988
Number of Answers: {n_ans}
Number of pred=Accept: {n_ac}
Number of pred=Reject: {n_rj}
Accuracy: {n_correct} / {n_ans} = {100 * n_correct / n_ans:.1f}%
Accuracy for big 4: {n_correct_pos} / {n_ans_pos} = {100 * n_correct_pos / n_ans_pos:.1f}%
Accuracy for arxiv: {n_correct_neg} / {n_ans_neg} = {100 * n_correct_neg / n_ans_neg:.1f}%
----- Latex table ----
{self.cfg.paper_type:10} & {n_ans:5} & {n_correct:5} & {100 * n_correct / n_ans:.1f}\\% & {n_ac} & {n_rj} & {100 * n_correct_pos / n_ans_pos:.1f}\\%  &  {100 * n_correct_neg / n_ans_neg:.1f}\\% \\\\
-----  End Latex  ----
''')


if __name__ == "__main__":
    # txt_analysis = GPTExpAnalysis(GPT_EXP_CFG_TXT, db_file="databases/ChatGPT_txt_May25.db")
    # txt_analysis.run()
    # print()
    # abs_analysis = GPTExpAnalysis(GPT_EXP_CFG_ABS, db_file="databases/ChatGPT_abs_May17.db")
    # abs_analysis.run()
    _run_queries(errors_only=False)
    # for d in data_list[:10]:
    #     print(d)
    # chunks = _test_chunks()
    # _first_abs_exp_results()
    # _debug()
    pass