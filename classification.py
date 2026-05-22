import time
from sys import stderr
from sys import exit as sys_exit

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB # , BernoulliNB, MultinomialNB, ComplementNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix #balanced_accuracy_score

# import config_conf
import config
from config import DatasetConfig, DatasetList
from config import ArxivDataset, SecDatasets, ClfDatasets

from utils import decision2score, score2binary
from Analyser import Analyser

pd.options.mode.chained_assignment = 'raise'

# def load_topics(dataset: config_conf.ConfDatasetConfig):
#     # if not dataset.filter_list:
#     #     sys_exit("Dataset with topics should have filter information.")
#     if dataset.topic != config_conf.NO_TOPIC:
#         df = pd.read_csv(dataset.topic, encoding="latin1")
#         df = df[['paper', 'topic']]
#         # df = df[df.topic != '<none>'] # drop none topics
#         df = df.groupby('paper')['topic'].apply(list)
#         topics = df[dataset.filter_list]
#         assert len(topics) == len(dataset.filter_list)
#         return topics.tolist()
#     return ['none'] * len(dataset.get_pp_txt_file_list())
#
#
# def load_score(dataset: config_conf.ConfDatasetConfig):
#     df = pd.read_csv(dataset.score, encoding="latin1")
#     df = df[['paper', 'decision']]
#     df = df.drop_duplicates()
#     df = df.loc[df.paper.isin(dataset.filter_list)]
#     decisions = df.decision.to_list()
#     labels = [decision2score(decision=d) for d in decisions]
#     return labels

# def train_val_split_separately(
#         vector_path = config.VECTOR_PATH,
#         datasets: DatasetList = config.ClfDatasets,
#         test_size = 0.2
# ):
#     print("Currently deprecated due to the implementation of scoring.", file=stderr, flush=True)
#     exit("Deprecated function used.")
#     y0 = np.array([1] * config_conf.LEN_SUBSET_1_SEC)
#     y1 = []
#     for DATASET in datasets:
#         if DATASET.score != config_conf.NO_SCORE:
#             y = load_score(DATASET)
#             print(f"{DATASET.name}: {y.count(1)}, {y.count(0)}, {len(y)}")
#             y1.extend(y)
#     y1 = np.array(y1)
#     X = np.load(vector_path)
#     X0, X1 = X[:config_conf.LEN_SUBSET_1_SEC], X[config_conf.LEN_SUBSET_1_SEC:]
#     assert len(y0) == X0.shape[0] and len(y1) == X1.shape[0]
#
#     X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, test_size = test_size, random_state = 42)
#     X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = test_size, random_state = 42)
#
#     X_train = np.append(X0_train, X1_train, axis=0)
#     y_train = np.append(y0_train, y1_train)
#
#     return X_train, (X0_test, X1_test), y_train, (y0_test, y1_test)

def load_data(
        vector_path = config.VECTOR_PATH,
        dataset_list: DatasetList | None = None,
        # n: 旧逻辑中假设「正样本数 = 负样本数 = n」，现在可能不再相等，保留为可选仅用于信息输出
        n: int | None = None,
):
    """
    根据向量文件和数据集配置构建带有标签的数据表：
    - 默认使用 config.AllDatasets（所有会议 + Arxiv）
    - 通过数据集名称区分正负样本：非 Arxiv 为正样本(1)，Arxiv 为负样本(0)
    """
    data_df = pd.DataFrame(np.load(vector_path))
    actual_size = data_df.shape[0]
    # 如果未显式指定数据集列表，默认使用「所有会议 + Arxiv」
    if dataset_list is None:
        dataset_list = config.ClfDatasets

    # 构建数据集名称列表，用于识别正负样本
    dataset_name = []
    for dataset in dataset_list:
        dataset_name.extend([dataset.name]*len(dataset))
    
    # 如果数据集名称数量不匹配，只使用实际向量数量
    if len(dataset_name) != actual_size:
        print(f"警告: 数据集名称数量 {len(dataset_name)} != 向量数量 {actual_size}，截断数据集名称")
        dataset_name = dataset_name[:actual_size]
    
    # 根据数据集名称来分配标签
    # 通常 ClfDatasets = ConfBig4Datasets + [ArxivDataset]
    # 正样本：ConfBig4Datasets（不是 ArxivDataset 的数据集）
    # 负样本：ArxivDataset
    data_df['dataset'] = dataset_name[:actual_size]
    
    # 识别 ArxivDataset（负样本），其他都是正样本
    # 检查最后一个数据集是否是 ArxivDataset
    arxiv_name = 'Arxiv_Kaggle_Data'  # ArxivDataset 的名称
    data_df['y'] = 1  # 默认都是正样本
    arxiv_mask = data_df['dataset'] == arxiv_name
    data_df.loc[arxiv_mask, 'y'] = 0  # ArxivDataset 是负样本
    
    # 统计实际的正负样本数量
    positive_count = (data_df['y'] == 1).sum()
    negative_count = (data_df['y'] == 0).sum()

    # n 为空时，只输出实际统计结果；否则保留兼容性提示
    if n is not None:
        expected_total = n * 2
        if actual_size != expected_total:
            print(f"警告: 向量数量 {actual_size} != 期望 {expected_total}，实际正样本: {positive_count}，负样本: {negative_count}")
        else:
            print(f"向量数量匹配，正样本: {positive_count}，负样本: {negative_count}")
    else:
        print(f"实际样本数量: 总计 {actual_size}，正样本: {positive_count}，负样本: {negative_count}")

    data_df['in_train_set'] = -1

    print(f"Total is {actual_size}, 1 : 0 = {positive_count} : {negative_count}", flush=True)
    return data_df

def load_data_RecentBig4(
        vector_path = config.VECTOR_PATH_RecentBig4,
        dataset_list: DatasetList = config.RecentBig4Datasets,
        # n: int = config.ArxivDataset_WORecentBig4.len_limit, # number of positive = number of negative = n
):
    data_df = pd.DataFrame(np.load(vector_path))
    # assert data_df.shape[0] == n
    dataset_name = []
    for dataset in dataset_list:
        dataset_name.extend([dataset.name]*len(dataset))
    assert len(dataset_name) == data_df.shape[0]
    data_df['y'] = 1 # all are positive (accept)
    data_df['in_train_set'] = 0
    data_df['dataset'] = dataset_name

    print(f"(RecentBig4) Total is {len(dataset_name)}.", flush=True)
    return data_df

def train_test_pipeline():
    if config.EXPERIMENT_TYPE == "Full":
        # Full 实验：使用所有会议作为正样本，Arxiv 作为负样本
        dataset_list = config.ClfDatasets
        vector_path_clf = config.VECTOR_PATH
        # 不再假设正负样本数量相等，因此不设置 n，避免无意义的长度告警
        n = None
    elif config.EXPERIMENT_TYPE == 'WORecentBig4':
        dataset_list = config.ClfDatasets_WORecentBig4
        vector_path_clf = config.VECTOR_PATH_WORecentBig4
        n = config.ArxivDataset_WORecentBig4.len_limit
    else:
        print("Experiment type error.")
        exit(1)


    print("The clf vector used in this training:", vector_path_clf)
    data_df = load_data(vector_path_clf, dataset_list, n)
    X_train, X_test, y_train, y_test = train_test_split(
        data_df[range(config.D2V_CONFIG['vector_size'])], data_df.y,
        test_size=0.2, random_state=42
    )
    data_df.loc[X_train.index, 'in_train_set'] = 1
    data_df.loc[X_test.index, 'in_train_set']  = 0
    print(f"train : val == {len(y_train)} : {len(y_test)}")
    print("y_test:", y_test.tolist(), flush=True)
    if config.EXPERIMENT_TYPE == 'WORecentBig4':
        recentbig4_df = load_data_RecentBig4()
        X_recentbig4 = recentbig4_df[range(config.D2V_CONFIG['vector_size'])]
    # print(len(y_test))
    # print(sum(y_test))
    # baseline_acc = (len(y_test) - sum(y_test)) / len(y_test)
    # print("The accuracy should be higher than", (len(y_test) - sum(y_test)), '/', len(y_test), '==', baseline_acc)

    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    classifiers = [
        ("VotingClassifier", VotingClassifier(estimators=[
            ("LogisticRegression", LogisticRegression(n_jobs=1, C=1.0, solver='newton-cg', class_weight='balanced')),
            ("RBF_SVM", SVC(gamma='scale', C=1e1)),
            ("GaussianProcess",
             GaussianProcessClassifier(C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)))),
            ("AdaBoost", AdaBoostClassifier(estimator=LogisticRegression(solver='lbfgs'), n_estimators=50)),
            ("LDA", LinearDiscriminantAnalysis()),
        ])),
        # ("VotingClassifier", VotingClassifier(estimators=[
        #     ("LogisticRegression", LogisticRegression(n_jobs=1, C=1e-3, solver='newton-cg', class_weight='balanced')),
        #     ("RBF_SVM", SVC(gamma='scale', C=1e3)),
        #     ("GaussianProcess",
        #      GaussianProcessClassifier(C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)))),
        #     ("AdaBoost", AdaBoostClassifier(estimator=LogisticRegression(solver='lbfgs'), n_estimators=50)),
        #     ("LDA", LinearDiscriminantAnalysis()),
        # ])),
        ("StackingClassifier", StackingClassifier(estimators=[
            ("LogisticRegression", LogisticRegression(n_jobs=1, C=1.0, solver='newton-cg', class_weight='balanced')),
            ("RBF_SVM", SVC(gamma='scale', C=1e1)),
            ("GaussianProcess",
             GaussianProcessClassifier(C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)))),
            ("AdaBoost", AdaBoostClassifier(estimator=LogisticRegression(solver='lbfgs'), n_estimators=50)),
            ("LDA", LinearDiscriminantAnalysis()),
        ], final_estimator=DecisionTreeClassifier())),
        ("LogisticRegression", LogisticRegression(n_jobs=1, C=1.0, solver='newton-cg', class_weight='balanced')),
        ("LinearSVM", SVC(kernel="linear", C=1.0)),
        ("RBF_SVM", SVC(gamma='scale', C=1e1)),
        ("PolySVM", SVC(C=1e1, kernel='poly', degree=2, gamma='scale')),
        ("GaussianProcess",
         GaussianProcessClassifier(C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)))),
        ("DecisionTree", DecisionTreeClassifier(criterion='entropy', max_depth=300)),
        ("RandomForest", RandomForestClassifier(n_estimators=1000, criterion='entropy', n_jobs=-1)),  # 使用所有CPU核心
        ("BoostedTree", GradientBoostingClassifier(max_depth=7, n_estimators=100, learning_rate=0.05)),
        # ("MLP", MLPClassifier(hidden_layer_sizes=(1000, 200, 10), activation='relu',
        #                       batch_size=200, learning_rate_init=0.001, learning_rate='adaptive',
        #                       solver='adam', max_iter=300, shuffle=True, )),
        ("MLP", MLPClassifier(hidden_layer_sizes=(512, 128), activation='relu',
                              batch_size=200, learning_rate_init=0.001, learning_rate='adaptive',
                              solver='adam', max_iter=300, shuffle=True, 
                              early_stopping=True, validation_fraction=0.1)), # 新增参数
        ("AdaBoost", AdaBoostClassifier(estimator=LogisticRegression(solver='lbfgs'), n_estimators=50)),
        ("GaussianNaiveBayes", GaussianNB()),
        ("KNN", KNeighborsClassifier(10)),
        ("LDA", LinearDiscriminantAnalysis()),
        ("QDA", QuadraticDiscriminantAnalysis()),
    ]

    latex_helper_str = ""
    statistics = {'acc':[], 'f1':[], 'auc':[], 'time':[], 'time_test':[]}
    # latex_sep = ""
    for name, clf in classifiers[:]: # change here to change how many classifiers to be used
        tic = time.time()
        print('-'*80)
        print(name + ':', flush=True)

        # Train
        clf.fit(X_train, y_train)
        data_df[f'pred_{name}'] = -1
        time_train = time.time() - tic

        # Predict
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        data_df.loc[X_train.index, f'pred_{name}'] = y_pred_train
        data_df.loc[X_test.index, f'pred_{name}'] = y_pred_test
        time_test = time.time() - tic - time_train

        # Report Evaluation Results
        print(f'    Training data accuracy is               : {accuracy_score(y_train, y_pred_train):.4f}')
        acc = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average='weighted')
        confusion_matrix_str = str(confusion_matrix(y_test, y_pred_test)).replace('\n', '\n'+' '*8)
        try:
            y_pred_prob = clf.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_pred_prob[:, 1])
        except AttributeError:
            auc = '-'
        print(f'''    Testing data accuracy overall is        : {acc:.4f}
    Testing F1 score overall                : {f1:.4f}
    Confusion matrix overall:
        {confusion_matrix_str}''')
        print('  AUC: ', auc, flush=True)
        print(f"  Train Time: {time_train:.2f} sec", flush=True)
        print(f"  Test Time: {time_test:.2f} sec", flush=True)
        statistics['acc'].append(acc)
        statistics['f1'].append(f1)
        statistics['time'].append(time_train)
        statistics['time_test'].append(time_test)
        if auc != '-':
            statistics['auc'].append(auc)

        if config.EXPERIMENT_TYPE == 'WORecentBig4':
            # Predict for recent big 4
            y_pred_recentbig4 = clf.predict(X_recentbig4)
            recentbig4_df.loc[X_recentbig4.index, f'pred_{name}'] = y_pred_recentbig4
            groups = recentbig4_df.groupby('dataset')
            acc_by_venue = {
                v: (g[f'pred_{name}'].sum() / len(g)) for v, g in groups
            }
            acc_venue_confs_template = "{:<8s} " * len(config.recent_big4_confs)
            acc_venue_accus_template = "{:.6f} " * len(config.recent_big4_confs)
            print(acc_venue_confs_template.format(*config.recent_big4_confs))
            print(acc_venue_accus_template.format(*(acc_by_venue[v] for v in config.recent_big4_confs)))

        # Latex helper string
        if auc == '-':
            latex_helper_str += f"{name:<20} & {acc:.3f} & {f1:.3f} & {auc:<5} & {time_train:.2f} & {time_test:.2f}\n"
        else:
            latex_helper_str += f"{name:<20} & {acc:.3f} & {f1:.3f} & {auc:.3f} & {time_train:.2f} & {time_test:.2f}\n"

    print('=' * 80, end='\n\n')
    print("----------------Latex helper for result------------------------")
    print(latex_helper_str)
    print('\n', flush=True)
    print( f"{'Average':<20} & {np.mean(statistics['acc']):.4f} & {np.mean(statistics['f1']):.4f} & {np.mean(statistics['auc']):.4f} & {np.mean(statistics['time']):.2f} & {np.mean(statistics['time_test']):.2f}")
    print( f"{'Median':<20} & {np.median(statistics['acc']):.4f} & {np.median(statistics['f1']):.4f} & {np.median(statistics['auc']):.4f} & {np.median(statistics['time']):.2f} & {np.mean(statistics['time_test']):.2f}")
    print("---------------------------------------------------------------")

    if config.EXPERIMENT_TYPE == 'WORecentBig4':
        return data_df, recentbig4_df
    else:
        return data_df

if __name__ == '__main__':
    # df = load_data(config.VECTOR_PATH, config.SECURITY_SET)
    df = train_test_pipeline()
    # analyser = Analyser(data_df)
    # analysis_report = analyser.analyse(by_score=True, by_topic=True)
    # analysis_report.to_csv(config_conf.REPORT_CSV_PATH)
    # visualizer = Visualizer(analysis_report)

    # To show postive:negtive, use the following code:
    # score_df = np.array(load_score(config.SUBSET_2[-1]))
    # print(f"{len(score_df)}: {np.sum(score_df>0)} : {np.sum(score_df<0)}")
