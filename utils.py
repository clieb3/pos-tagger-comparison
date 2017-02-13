import itertools
import random
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def num_tokens(tagged_sents):
    return sum([len(s) for s in tagged_sents])

def make_taggers(tagger_classes, num_train_baseline):
    return [c(train_size=(i * num_train_baseline))
        for i, c in itertools.product(range(1, 6), tagger_classes)]

def train_tagger(tagger, train_sents):
        t0 = time.time()
        tagger.train(train_sents)
        train_time = time.time() - t0
#         print ("{:20}{:3.2f}".format(type(tagger).__name__, train_time))
        return train_time

def train_taggers(taggers, train_sents):
    return [train_tagger(t, train_sents) for t in taggers]

def test_tagger(tagger, test_sents, num_tests):
    t0 = time.time()
    accuracy = tagger.test(test_sents[:num_tests])
    test_time = time.time() - t0
#     print ("{:20}{:3.2f}".format(type(tagger).__name__, test_time))
    return accuracy, test_time

def test_taggers(taggers, test_sents, num_tests):
    return map(list, zip(*[test_tagger(t, test_sents, num_tests) for t in taggers]))

def my_plot(grouped, field):
    fig, ax = plt.subplots(figsize=(8,6))
    for key, grp in grouped:
        grp.plot(x='tokens_size', y=field, ax=ax, label=key)
    plt.legend(loc='best')
    plt.title(field)
    plt.show()

def my_plots(grouped, fields):
    for field in fields:
        my_plot(grouped, field)

def compare_taggers(tagger_classes, train_sents, test_sents, num_train_baseline, num_tests):
    taggers = make_taggers(tagger_classes, num_train_baseline)
    train_times = train_taggers(taggers, train_sents)
    accuracies, test_times = test_taggers(taggers, test_sents, num_tests)

    tagger_types = [type(t).__name__ for t in taggers]
    tokens_sizes = [t.tokens_size for t in taggers]

    test_runs = [('tagger_type', tagger_types),
                 ('tokens_size', tokens_sizes),
                 ('accuracy', accuracies),
                 ('train_time', train_times),
                 ('test_time', test_times)]

    df = pd.DataFrame.from_items(test_runs)
    grouped = df.groupby(['tagger_type'])

    my_plots(grouped, ['accuracy', 'train_time', 'test_time'])

    return grouped


