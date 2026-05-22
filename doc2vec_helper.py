from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
# import smart_open
import os
import sys
from datetime import datetime

import config
from config import D2V_CONFIG, DatasetList
import utils


def construct_dataset(datasetlist: DatasetList, idx = 0, doc_type='txt', DEBUG=True):
    '''construct docs from datasetlist'''
    docs = []
    # idx = 0
    assert doc_type in ['txt', 'abs']
    for dataset in datasetlist:
        if doc_type == 'txt':
            pp_dir = dataset.txt_pp_dir
            file_list = dataset.get_pp_txt_file_list()
        else:
            pp_dir = dataset.abs_pp_dir
            file_list = dataset.get_pp_abs_file_list()
        print(f"  Constructing {dataset.name} from {pp_dir}:", end='')
        print(f"{len(file_list)} files.")
        
        # 记录本数据集开始时的文档数量
        start_docs_count = len(docs)
        skipped = 0
        
        for file_name in file_list:
            if not os.path.exists(file_name):
                skipped += 1
                if skipped <= 5:  # 只打印前5个缺失的文件
                    print(f"  Warning: Skipping missing file {file_name}")
                continue
            try:
                with open(file_name) as f:
                    txt = f.read()
                docs.append(TaggedDocument(words=txt.split(), tags=[idx]))
                idx += 1
            except Exception as e:
                skipped += 1
                print(f"  Warning: Error reading file {file_name}: {e}")
                continue
            if DEBUG and idx>5:
                break
        
        # 计算本数据集实际处理的文件数
        actual_processed = len(docs) - start_docs_count
        # if skipped > 0:
            # print(f"  跳过 {skipped} 个缺失或错误的文件，实际处理 {actual_processed} 个文件")
    return docs, idx

def doc2vec_train(docs_all, docs_sec):
    '''train a doc2vec model on given docs_all'''
    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
    print('Start training process...', time_str)
    sys.stdout.flush()

    model = Doc2Vec(**D2V_CONFIG)
    model.build_vocab(docs_all)
    # model.train(docs_all, total_examples=model.corpus_count, epochs=model.epochs)
    model.train(docs_all, total_examples=len(docs_all), epochs=model.epochs)
    print("Fine-tuning time: ", datetime.now().strftime("%Y-%m-%d-%H-%M"))
    model.train(docs_sec, total_examples=len(docs_sec), epochs=20, start_alpha=0.002, end_alpha=0.001)
    print("End time: ", datetime.now().strftime("%Y-%m-%d-%H-%M"))

    # save model
    model_name = f"{config.DOC_TYPE}_{config.DOC2VEC_TYPE}_{config.EXPERIMENT_TYPE}_{time_str}.model"
    model.save(os.path.join(config.MODEL_PATH, model_name))
    print("Model is saved as:", model_name)

    # 尝试删除临时训练数据（如果方法存在）
    if hasattr(model, 'delete_temporary_training_data'):
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    return model

# txt2list = lambda txt: txt.split()

def doc2vec_infer(model, docs, dest = config.VECTOR_PATH):
    '''docs should be a list of TaggedDocument'''
    n = len(docs)
    vecs = np.zeros((n, D2V_CONFIG['vector_size']))
    for i, doc in enumerate(docs):
        vecs[i] = np.array(model.infer_vector(doc.words))
    np.save(dest, vecs)
    print("Vectors file is saved to:", dest)

if __name__ == '__main__':
    print(f"Doc2Vec for experiment type {config.EXPERIMENT_TYPE}.")
    print(f"Running Doc2Vec for {config.DOC_TYPE}, the model is {config.DOC2VEC_TYPE}:")

    docs_all, idx = construct_dataset(config.AllDatasets, doc_type=config.DOC_TYPE, DEBUG=False)
    docs_sec, idx = construct_dataset(config.SecDatasets, idx=idx, doc_type=config.DOC_TYPE, DEBUG=False)
    model = doc2vec_train(docs_all=docs_all, docs_sec=docs_sec)
    if config.EXPERIMENT_TYPE == "Full":
        docs_clf, idx = construct_dataset(config.ClfDatasets, idx=idx, doc_type=config.DOC_TYPE, DEBUG=False)
        print(len(docs_all), len(docs_sec), len(docs_clf))
        doc2vec_infer(model, docs_clf)
    elif config.EXPERIMENT_TYPE == 'WORecentBig4':
        docs_clf_WO, idx = construct_dataset(config.ClfDatasets_WORecentBig4, idx=idx, doc_type=config.DOC_TYPE, DEBUG=False)
        docs_recentbig4, idx = construct_dataset(config.RecentBig4Datasets, idx=idx, doc_type=config.DOC_TYPE, DEBUG=False)
        print(len(docs_all), len(docs_sec), len(docs_clf_WO), len(docs_recentbig4))
        doc2vec_infer(model, docs_clf_WO, dest=config.VECTOR_PATH_WORecentBig4)
        doc2vec_infer(model, docs_recentbig4, dest=config.VECTOR_PATH_RecentBig4)
    elif config.EXPERIMENT_TYPE == 'Both':
        docs_clf, idx = construct_dataset(config.ClfDatasets, idx=idx, doc_type=config.DOC_TYPE, DEBUG=False)
        docs_clf_WO, idx = construct_dataset(config.ClfDatasets_WORecentBig4, idx=idx, doc_type=config.DOC_TYPE, DEBUG=False)
        docs_recentbig4, idx = construct_dataset(config.RecentBig4Datasets, idx=idx, doc_type=config.DOC_TYPE, DEBUG=False)
        print(len(docs_all), len(docs_sec))
        print(len(docs_clf))
        print(len(docs_clf_WO), len(docs_recentbig4))
        doc2vec_infer(model, docs_clf, dest=config.VECTOR_PATH)
        doc2vec_infer(model, docs_clf_WO, dest=config.VECTOR_PATH_WORecentBig4)
        doc2vec_infer(model, docs_recentbig4, dest=config.VECTOR_PATH_RecentBig4)
    print("Done.")
