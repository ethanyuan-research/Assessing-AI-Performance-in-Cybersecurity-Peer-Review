import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix

from utils import score2decision, aggregated_decisions, deprecated
# from archives.archived.topics import topics_little2big, topics_big2little, general_topics
# 临时定义，避免导入错误
topics_little2big = {}
topics_big2little = {}
general_topics = []

class Analyser:

    def __init__(self, df: pd.DataFrame):
        self.dataframe = df
        self.cols_pred = [col for col in df.columns if str(col).startswith('pred_')]
        self.cols_wo_vp = [c for c in df.columns if
                           not isinstance(c, int) and not str(c).startswith('pred_')]  # without pred/vec
        self.df_test = df[self.cols_wo_vp + self.cols_pred].loc[df['in_train_set'] == 0]

    def _accuracy(self, df_pred, idx_cond, col):
        df_condition_pred = df_pred[col].loc[idx_cond]
        correct_pred = (df_condition_pred == df_pred.y.loc[idx_cond]).sum()
        acc = correct_pred / len(df_condition_pred)
        return correct_pred, len(df_condition_pred), acc

    def _confusion_matrix(self, df_pred, idx_cond, col):
        # TODO: to be tested
        df_condition_pred = df_pred[col].loc[idx_cond]
        TN, FP, FN, TP = confusion_matrix(df_pred.y.loc[idx_cond], df_condition_pred).ravel()
        return TN, FP, FN, TP

    def _generate_report_columns(self):
        report_columns = list(map(lambda x:x.split('_')[1], self.cols_pred))

    def _generate_report_index(self, decision_list, topic_list):
        report_index_score, report_index_topic = [], []
        if decision_list:
            product_list_score = itertools.product(decision_list, ["TN", "FP", "FN", "TP"])
            report_index_score = list(map(lambda x: ' -- '.join(x), product_list_score))
        if topic_list:
            product_list_topic = itertools.product(topic_list, ["TN", "FP", "FN", "TP"])
            report_index_topic = list(map(lambda x: ' -- '.join(x), product_list_topic))
        return report_index_score + report_index_topic

    def analyse(self, by_score: bool = True, by_topic: bool = True):
        analysis_report = []
        analysis_report_df = pd.DataFrame(index=self._generate_report_index(decision_list=aggregated_decisions,
                                                                            topic_list=general_topics),
                                          columns=self._generate_report_columns())
        for col in self.cols_pred:
            clf_name = col.split('_')[1]
            print(f"{clf_name}:")
            df_pred = self.df_test[self.cols_wo_vp + [col]]

            # analyse by score
            if by_score:
                print("  Analyze by score:\n")
                for score in np.sort(df_pred.score.unique()):
                    idx_cond = df_pred.score == score
                    num_correct, num_all, acc = self._accuracy(df_pred, idx_cond, col)
                    analysis_report.append((num_correct, num_all, acc, clf_name, 'by_score', score2decision(score)))
                    conf_mat = self._confusion_matrix(df_pred, idx_cond, col)
                    analysis_report_df.loc[self._generate_report_index(decision_list=[score2decision(score)],
                                                                       topic_list=None), clf_name] = conf_mat
                    print(f'    {score2decision(score):44} papers accuracy: {num_correct:3} / {num_all:3} = {acc:.3f}')
                    print(f'      TN, FP, FN, TP: {conf_mat}')
                print('')

            # analyse by topic
            if by_topic:
                print("  Analyze by topic:\n")
                df_pred = df_pred[df_pred.topics != 'none']
                df_pred['topics'] = df_pred['topics'].apply(lambda l: list(set(
                    map(lambda t: topics_little2big[t], l)
                )))
                for general_topic in topics_big2little:
                    if general_topic == 'none':
                        continue
                    idx_cond = df_pred.topics.apply(lambda topics: general_topic in topics)
                    num_correct, num_all, acc = self._accuracy(df_pred, idx_cond, col)
                    conf_mat = self._confusion_matrix(df_pred, idx_cond, col)
                    analysis_report.append((num_correct, num_all, acc, clf_name, 'by_topic', general_topic))
                    analysis_report_df.loc[self._generate_report_index(topic_list=[general_topic],
                                                                       decision_list=None), clf_name] = conf_mat
                    print(f'    {general_topic:44} papers accuracy: {num_correct:3} / {num_all:3} = {acc:.3f}')
                    print(f'      TN, FP, FN, TP: {conf_mat}')
                print('-' * 80)
        return analysis_report_df

@deprecated
def analyze_by_score(df: pd.DataFrame):
    cols_pred = [col for col in df.columns if str(col).startswith('pred_')]
    cols_wo_vp = [c for c in df.columns if
                  not isinstance(c, int) and not str(c).startswith('pred_')]  # without pred/vec
    df_test = df[cols_wo_vp + cols_pred].loc[df['in_train_set'] == 0]
    print("Analyze by score:\n")
    for col in cols_pred:
        clf_name = col.split('_')[1]
        print(f"{clf_name}:")
        df_pred = df_test[cols_wo_vp + [col]]
        for score in np.sort(df_pred.score.unique()):
            df_score_pred = df_pred[col].loc[df_pred.score == score]

            # ac_ratio = df_score_pred.sum() / len(df_score_pred)
            # print(f'{score2decision(score):44} papers acceptance ratio: {df_score_pred.sum():3} / {len(df_score_pred):3} = {ac_ratio:.3f}')
            correct_pred = (df_score_pred == df_pred.y.loc[df_pred.score == score]).sum()
            acc = correct_pred / len(df_score_pred)
            print(
                f'  {score2decision(score):44} papers predict accuracy: {correct_pred:3} / {len(df_score_pred):3} = {acc:.3f}')
        print('')
        # return df_pred


if __name__ == '__main__':
    pass
