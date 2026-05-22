import os
import re
import sqlite3
from math import radians, tan
from sys import stderr
import numpy as np
from sklearn.model_selection import train_test_split

import warnings
import functools

# https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically
def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


def get_gpt_abs_examples():
    return [
        '''Wireless Sensor Networks (WSNs) rely on in-
    network aggregation for efﬁciency, however, this comes at a
    price: A single adversary can severely inﬂuence the outcome
    by contributing an arbitrary partial aggregate value. Secure
    in-network aggregation can detect such manipulation [2]. But
    as long as such faults persist, no aggregation result can be
    obtained. In contrast, the collection of individual sensor node
    values is robust and solves the problem of availability, yet in
    an inefﬁcient way. Our work seeks to bridge this gap in secure
    data collection: We propose a system that enhances availability
    with an efﬁciency close to that of in-network aggregation. To
    achieve this, our scheme relies on costly operations to localize and
    exclude nodes that manipulate the aggregation, but only when a
    failure is detected. The detection of aggregation disruptions and
    the removal of faulty nodes provides robustness. At the same
    time, after removing faulty nodes, the WSN can enjoy low cost
    (secure) aggregation. Thus, the high exclusion cost is amortized,
    and efﬁciency increases.''',  # from arxiv
        '''Serverless computing has freed developers from the burden
    of managing their own platform and infrastructure, allowing
    them to rapidly prototype and deploy applications. Despite
    its surging popularity, however, serverless raises a number of
    concerning security implications. Among them is the difﬁ-
    culty of investigating intrusions – by decomposing traditional
    applications into ephemeral re-entrant functions, serverless
    has enabled attackers to conceal their activities within legit-
    imate workﬂows, and even prevent root cause analysis by
    abusing warm container reuse policies to break causal paths.
    Unfortunately, neither traditional approaches to system audit-
    ing nor commercial serverless security products provide the
    transparency needed to accurately track these novel threats.
    In this work, we propose ALASTOR, a provenance-based
    auditing framework that enables precise tracing of suspicious
    events in serverless applications. ALASTOR records function
    activity at both system and application layers to capture a
    holistic picture of each function instances’ behavior. It then
    aggregates provenance from different functions at a central
    repository within the serverless platform, stitching it together
    to produce a global data provenance graph of complex func-
    tion workﬂows. ALASTOR is both function and language-
    agnostic, and can easily be integrated into existing serverless
    platforms with minimal modiﬁcation. We implement ALAS-
    TOR for the OpenFaaS platform and evaluate its performance
    using the well-established Nordstrom Hello,Retail! applica-
    tion, discovering in the process that ALASTOR imposes man-
    ageable overheads (13.74%), in exchange for signiﬁcantly
    improved forensic capabilities as compared to commercially-
    available monitoring tools. To our knowledge, ALASTOR is
    the ﬁrst auditing framework speciﬁcally designed to satisfy
    the operational requirements of serverless platforms. ''',  # from usenix
    ]


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def auc(x, y):
    area = 0
    last_x = 0
    for x,y in zip(x,y):
        area += y*(x-last_x)
        last_x = x
    return area

def train_test_split_with_score(x: np.ndarray, y: np.ndarray, score = None, **args):
    print("This funciton is now deprecated because of DataFrame is now used for data.", file=stderr, flush=True)
    exit("Deprecated function 'train_test_split_with_score'.")
    assert np.ndim(y) == 1 and np.ndim(x) == 2
    if score is not None:
        assert y.shape[0] == x.shape[0] and score.shape[0] == x.shape[0]
        if np.ndim(score) == 1:
            score = score[:, np.newaxis]
        y = np.hstack((y[:, np.newaxis], score))
    return train_test_split(x, y, **args)

def papername_to_number(name):
    name = name.split('.')[0]
    return int(re.findall(r'\d+', name)[-1])

def height2others():
    pass


decision_score_mapping = {
    'early rejected': -2,# -4: 'early rejected',
    'round 1 reject': -2,# -3: 'round 1 reject',
    'rejected': -2,# -2: 'rejected',
    'reject and resubmit': -1,# -1: 'reject and resubmit',
    'major revision': -1,# 0: 'major revision',
    'accept shepherd': 1,# 1: 'accept shepherd',
    'accepted with shepherding': 1,# 2: 'accepted with shepherding',
    'minor revision': 1,# 3: 'minor revision',
    'accepted': 4,# 4: 'accepted',
    'published': 10,# 10: 'published',

    -2: 'rejected / early reject / round 1 reject',
    -1: 'reject and resubmit / major revision',
    1: 'accepted with shepherding / minor revision',
    4: 'accepted',
    10: 'published',
}

aggregated_decisions = ['rejected / early reject / round 1 reject',
                        'reject and resubmit / major revision',
                        'accepted with shepherding / minor revision',
                        'accepted',
                        'published', ]

def decision2score(decision: str):
    return decision_score_mapping[decision.lower()]

def score2decision(score):
    return decision_score_mapping[score]

def score2binary(score):
    return np.where(score>0, 1, 0)

def latex_avg_median(s: str):
    from statistics import median, mean
    data = s.split()
    data = list(map(float, data))
    print(mean(data), median(data))

if __name__ == '__main__':
    pass
    # x = np.diag(range(1,6))
    # print(x)
    # y = np.array([-1,-2,-3,-4,-5])
    # print(y)
    # score = np.array([10,20,30,40,50])
    # print(score)
    # print('-'*80)
    #
    # x_train, x_test, y_train, y_test = train_test_split_with_score(x,y,score, test_size=0.2)
    # print(papername_to_number('sec18-paper8.pdf'), papername_to_number('ndss18-paper267.pdf'))
    print("accuracy")
    latex_avg_median("0.8602 0.8241 0.8534 0.8568 0.8658 0.8591 0.7227 0.7035 0.8557 0.8388 0.8501 0.8399 0.8230 0.5738 0.8613 0.8433")
    print("f1")
    latex_avg_median("0.8600 0.8241 0.8531 0.8563 0.8656 0.8591 0.7120 0.7031 0.8543 0.8382 0.8500 0.8398 0.8204 0.4931 0.8609 0.8404")
    print("auc")
    latex_avg_median("0.8240 0.9251 0.5034 0.7042 0.9156 0.9000 0.9145 0.9160 0.8725 0.8343 0.9233 0.9006")
    print("time")
    latex_avg_median("90.07  579.04 0.13   3.61   7.20   7.32   83.46  2.04   10.33  61.35  15.30  0.71   0.07   8.38   0.16   0.13")

