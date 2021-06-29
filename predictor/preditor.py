import tempfile
from typing import Dict, Iterable, List, Tuple
from overrides import overrides
import os
import json
from tqdm import tqdm
import glob
import re
from sklearn.model_selection import train_test_split

import torch

from allennlp.common.util import JsonDict
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.fields import LabelField, TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerEmbedder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util
from allennlp.predictors import Predictor, TextClassifierPredictor
from allennlp.training.metrics import CategoricalAccuracy, SequenceAccuracy, Metric, F1Measure
from allennlp.training.optimizers import AdamOptimizer, HuggingfaceAdamWOptimizer
from allennlp.training.trainer import Trainer
from allennlp.training import GradientDescentTrainer
from allennlp.training.util import evaluate
from allennlp.data.data_loaders import MultiProcessDataLoader
from model.model import PuncRestoreLabeler
import re
import argparse

parser = argparse.ArgumentParser(description='arg of model.')
parser.add_argument('--model_name', type=str, default="hfl/chinese-bert-wwm")

args = parser.parse_args()


class PuncPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)

def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    embedder = PretrainedTransformerEmbedder(model_name)
    encoder = BasicTextFieldEmbedder(token_embedders={'tokens': embedder})
    return PuncRestoreLabeler(vocab, embedder, encoder)


def get_punc(query, predictor):
    pred = predictor.predict(query)['probs'][1:-1]
    pun_sentence = [token + '，' if tag == 1 else token for token, tag in zip(list(query), pred)]
    #     print(pun_sentence)
    if pun_sentence[-1][-1] == '，':
        pun_sentence[-1] = pun_sentence[-1].replace('，', '。')
    else:
        pun_sentence.append('。')

    return ''.join(pun_sentence)


def test_punc(sentence):
    r = "[，。；？！]"
    no_punc_sentence = re.sub(r, '', sentence)
    #     print(no_punc_sentence)
    punc_sentence = get_punc(no_punc_sentence, predictor)

    print(sentence, punc_sentence, sep='\n')

if __name__ == '__main__':
    model_name = args.model_name
    model = build_model()
    predictor = PuncPredictor()
    test_punc(sentence)