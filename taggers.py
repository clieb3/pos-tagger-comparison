import nltk
from utils import num_tokens

def default_backoff_tagger(train_sents):
    tags = [tag for sent in train_sents for (word, tag) in sent]
    nltk.DefaultTagger(nltk.FreqDist(tags).max())

class UnigramTaggerBackoff(object):
    def __init__(self, train_size):
        self.train_size = train_size

    def train(self, train_sents):
        train_sents = train_sents[:self.train_size]
        self.tokens_size = num_tokens(train_sents)

        t0 = default_backoff_tagger(train_sents)

        self._tagger = nltk.tag.UnigramTagger(train_sents, backoff=t0)

    def test(self, test_sents):
        return self._tagger.evaluate(test_sents)

class BigramTaggerBackoff(object):
    def __init__(self, train_size):
        self.train_size = train_size

    def train(self, train_sents):
        train_sents = train_sents[:self.train_size]
        self.tokens_size = num_tokens(train_sents)

        t0 = default_backoff_tagger(train_sents)
        t1 = nltk.tag.UnigramTagger(train_sents, backoff=t0)

        self._tagger = nltk.tag.BigramTagger(train_sents, backoff=t1)

    def test(self, test_sents):
        return self._tagger.evaluate(test_sents)

class TrigramTaggerBackoff(object):
    def __init__(self, train_size):
        self.train_size = train_size

    def train(self, train_sents):
        train_sents = train_sents[:self.train_size]
        self.tokens_size = num_tokens(train_sents)

        t0 = default_backoff_tagger(train_sents)
        t1 = nltk.tag.UnigramTagger(train_sents, backoff=t0)
        t2 = nltk.tag.BigramTagger(train_sents, backoff=t1)

        self._tagger = nltk.tag.TrigramTagger(train_sents, backoff=t2)

    def test(self, test_sents):
        return self._tagger.evaluate(test_sents)

class TnTTaggerBackoff(object):
    def __init__(self, train_size):
        self.train_size = train_size

    def train(self, train_sents):
        train_sents = train_sents[:self.train_size]
        self.tokens_size = num_tokens(train_sents)

        t0 = default_backoff_tagger(train_sents)

        self._tagger = nltk.tag.tnt.TnT(unk=t0)
        self._tagger.train(train_sents)

    def test(self, test_sents):
        return self._tagger.evaluate(test_sents)

class BigramTaggerDefaultBackoff(object):
    def __init__(self, train_size):
        self.train_size = train_size

    def train(self, train_sents):
        train_sents = train_sents[:self.train_size]
        self.tokens_size = num_tokens(train_sents)

        t0 = default_backoff_tagger(train_sents)

        self._tagger = nltk.tag.BigramTagger(train_sents, backoff=t0)

    def test(self, test_sents):
        return self._tagger.evaluate(test_sents)

class TrigramTaggerDefaultBackoff(object):
    def __init__(self, train_size):
        self.train_size = train_size

    def train(self, train_sents):
        train_sents = train_sents[:self.train_size]
        self.tokens_size = num_tokens(train_sents)

        t0 = default_backoff_tagger(train_sents)

        self._tagger = nltk.tag.TrigramTagger(train_sents, backoff=t0)

    def test(self, test_sents):
        return self._tagger.evaluate(test_sents)
