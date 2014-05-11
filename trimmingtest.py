from nlpio import *
from trimming import *
from nlplearn import *
from sklearn.pipeline import Pipeline
from sklearn.grid_search import ParameterGrid
import pickle
import os.path


if __name__ == '__main__':

    filename = 'testset'
    if os.path.isfile('.'.join([filename, 'pk'])):
        # Already parsed by stanford corenlp
        documents = pickle.load(open('.'.join([filename, 'pk'])))
    else:
        # Brace yourself
        documents = loadDocumentsFromFile('.'.join([filename, 'txt']))

    # Compressor trained with the rest of duc2004
    sc = pickle.load(open('trainedcompressor.pk'))

    pipeline = Pipeline([
        ('clean', SimpleTextCleaner()),
        ('sentence', SentenceSplitter()),
        ('parse', StanfordParser()),
        ('compress', sc),
        ('select', SentenceSelector()),
        ])

    parameters = {
        'compress__tags_importance':
        [i/10.0 for i in xrange(1,10)],
    }

    # scorer = RougeScorer()

    # grid = ParameterGrid(parameters)
    # gridpoints = list(grid)

    # best_score = 0.0
    # best_point = None
    # for i, point in enumerate(gridpoints):
    #     pipeline.set_params(**point)
    #     score = scorer(pipeline, documents)
    #     if score > best_score:
    #         best_score = score
    #         best_point = point
    #     print score
    #     print point
    #     print "Iteration %i/%i" % (i+1, len(gridpoints))

    print pipeline.predict(documents)

    for s in documents[0].ext['compressed_sentences'][0]:
        print s