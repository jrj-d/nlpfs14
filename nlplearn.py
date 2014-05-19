from nlpio import *

import numpy as np
import logging
import random
import nltk.data

from sklearn.base import BaseEstimator,TransformerMixin

class RougeScorer(object):
    #variant can be any rouge variant available in the output
    #metric should be one of P,R,F
    def __init__(self,variant='ROUGE-1',metric='F'):
        self.variant = variant
        self.metric = metric

    def __call__(self,estimator,documents,predictions=None):
        if predictions is None:
            predictions = estimator.predict(documents)
        results = evaluateRouge(documents,predictions)
        return results[self.variant][self.metric][0]

class HeadlineEstimator(BaseEstimator):
    '''Estimates, for a given document, its headline'''
    def __init__(self):
        pass

    def fit(self, documents, y=None): #we generate the headlines directly from the documents, so we don't need a "y"
        return self

    def predict(self, documents):
        return [doc.ext['sentences'][0] for doc in documents] #for now this just predicts the first sentence

class SimpleTextCleaner(BaseEstimator,TransformerMixin):
    #TODO: make better
    def __init__(self):
        pass

    def fit(self,documents,y=None):
        return self

    def transform(self,documents):
        for doc in documents:
            doc.text = re.sub("`|'|\"","",doc.text)
            doc.text = re.sub("(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\.","\\1",doc.text)
        return documents

trained_splitter = None

class SentenceSplitter(BaseEstimator,TransformerMixin):
    #TODO: make better
    def __init__(self):
        global trained_splitter
        if trained_splitter is None:
            trained_splitter = nltk.data.load('tokenizers/punkt/english.pickle')

    def fit(self,documents,y=None):
        return self

    def transform(self,documents):
        for doc in documents:
            if not 'sentences' in doc.ext:
                doc.ext['sentences'] = trained_splitter.tokenize(doc.text.strip())
        return documents


class PeerEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, peer=0):
        self.peer = peer

    def fit(self, documents, y=None):
        return self

    def predict(self, documents):
        return [doc.peers[self.peer] for doc in documents]
