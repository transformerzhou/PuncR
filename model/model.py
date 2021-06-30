import tempfile
from typing import Dict, Iterable, List, Tuple
from overrides import overrides
import torch

from allennlp.common.util import JsonDict
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.training.metrics import CategoricalAccuracy, SequenceAccuracy, Metric



class F1(Metric):
    def __init__(self):
        self.correct_num, self.predict_num, self.gold_num = 1e-10, 1e-10, 1e-10

    def __call__(self, probs, label):
        #         print(probs1.shape, label1.shape)
        self.correct_num += torch.sum((probs == 1) & (label == 1)).cpu().numpy()
        self.predict_num += torch.sum(probs == 1).cpu().numpy()
        self.gold_num += torch.sum(label == 1).cpu().numpy()

    def get_metric(self, reset: bool = False) -> Tuple[float, float, float]:
        precision = self.correct_num / self.predict_num
        recall = self.correct_num / self.gold_num
        f1_score = 2 * precision * recall / (precision + recall)

        if reset:
            self.reset()
        return {'pre': precision, 'rec': recall, 'f1': f1_score}

    @overrides
    def reset(self):
        print("correct_num：{}，gold_num：{}， predict_num：{}".format(self.correct_num, self.gold_num, self.predict_num))
        self.correct_num, self.predict_num, self.gold_num = 1e-10, 1e-10, 1e-10

@Model.register("tagger")
class PuncRestoreLabeler(Model):
    def __init__(
            self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(self.encoder.get_output_dim(), 1)
        self.f1 = F1()

    #         self.f1 = F1Measure(1)

    def forward(
            self,
            text: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        mask = text['tokens']['mask']
        # Shape: (batch_size, num_tokens, embedding_dim)
        encoded_text = self.encoder(text)
        logits1 = self.classifier(encoded_text)

        logits2 = torch.squeeze(logits1, dim=-1)
        probs = torch.sigmoid(logits2)
        th = 0.95
        probs[torch.where(probs > th)] = 1
        probs[torch.where(probs <= th)] = 0
        output = {"probs": probs}
        if label is not None:
            self.f1(probs, label)
            output["loss"] = (torch.nn.functional.binary_cross_entropy_with_logits(logits2, label.float(),
                                                                                   reduction='none') * mask).sum() / (
                                         mask.sum() + 1e-9)
        #             output["loss"] = (torch.nn.functional.binary_cross_entropy_with_logits(logits2, label))
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.f1.get_metric(reset)