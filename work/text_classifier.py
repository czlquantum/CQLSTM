from itertools import chain
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from allennlp.data import TextFieldTensors
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, FBetaMultiLabelMeasure, FBetaMeasure, BooleanAccuracy
from allennlp.training import GradientDescentTrainer
# from allennlp_models.classification.dataset_readers.stanford_sentiment_tree_bank import \
#     StanfordSentimentTreeBankDatasetReader

# from realworldnlp.predictors import SentenceClassifierPredictor

EMBEDDING_DIM = 128
HIDDEN_DIM = 128

@Model.register("complex_text_classifier")
class ComplexTextClassifier(Model):

    def __init__(self,
                 embedder_real: TextFieldEmbedder,
                 embedder_imag: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 pooler: Seq2VecEncoder,
                 vocab: Vocabulary,
                 num_classes: int = 2,
                 positive_label: str = '1') -> None:
        super().__init__(vocab)
        self.embedder_real = embedder_real
        self.embedder_imag = embedder_imag
        self.encoder = encoder
        self.pooler = pooler
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim() + pooler.get_output_dim(),
                                      out_features=self.num_classes)
        positive_index = vocab.get_token_index(positive_label, namespace='label')

        # voc = vocab.get_index_to_token_vocabulary()

        # self.accuracy = CategoricalAccuracy()
        # self.f1_measure = F1Measure(positive_index)
        self.bool_acc = BooleanAccuracy()
        self.f1_multi = FBetaMultiLabelMeasure(average='macro')
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self,
                tokens: TextFieldTensors,
                label: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        

        embeddings_real = self.embedder_real(tokens)
        embeddings_imag = self.embedder_imag(tokens)
        embeddings = embeddings_real + 1j * embeddings_imag
        encoder_out = self.encoder(embeddings, mask)
        pooler_out = self.pooler(embeddings_real, mask)
        # logits = self.linear(encoder_out) 沿着最后一个维度进行拼接
        logits = self.linear(torch.cat([encoder_out, pooler_out], dim=-1))
        probs = torch.softmax(logits, dim=-1)
        output = {"logits": logits, "cls_emb": encoder_out, "probs": probs}
        preds = torch.argmax(logits, dim=-1)
        
        if label is not None:
            self.bool_acc(preds, label)
            one_hot_preds = F.one_hot(preds, self.num_classes)
            one_hot_labels = F.one_hot(label, self.num_classes)
            try:
                self.f1_multi(one_hot_preds, one_hot_labels)
            except:
                # print(preds)
                # print(label)
                self.f1_multi(one_hot_preds, one_hot_labels)
                # a = 1/ 0
            # self.accuracy(logits, label)
            # self.f1_measure(logits, label)
            output["loss"] = self.loss_function(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                # 'accuracy': self.accuracy.get_metric(reset),
                # **self.f1_measure.get_metric(reset)
                'bool_accuracy': self.bool_acc.get_metric(reset),
                **self.f1_multi.get_metric(reset)
                }
