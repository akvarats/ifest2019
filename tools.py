import os
import nltk
import gensim

from string import punctuation
from rnnmorph.predictor import RNNMorphPredictor


def prepare_text(text):
    """
    """

    words = [w for w in nltk.word_tokenize(text, language="russian") if w not in punctuation]

    predictor = RNNMorphPredictor(language="ru")
    morphs = predictor.predict(words)

    return ["{}_{}".format(m.normal_form, m.pos) for m in morphs]


def get_corpus():
    return [
        "Не горит фонарь уличного освещения над пешеходным переходом",
        "На протяжении нескольких лет отсутствие горячего водоснабжения в селе которое раньше было",
        "На перекрестке улицы Юбилейной и Московском шоссе выдран светофор из-за ДТП",
    ]


def prepare_corpus(corpus):
    return [
        prepare_text(text) for text in corpus
    ]

    
def get_word2vec_model():
    """
    """
    model_path = os.path.join(os.path.dirname(__file__), "models/180/model.bin")
    return gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)


