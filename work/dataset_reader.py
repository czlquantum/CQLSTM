# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  dataset.py
@Time    :  2023/05/20
@Author  :  chu
@Contact :  chuzlu123@163.com
@Desc    :  None
"""
from typing import Dict, List, Union
import logging
import json

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter

logger = logging.getLogger(__name__)


@DatasetReader.register("my_dataset_reader")
class TextClassificationReader(DatasetReader):

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Tokenizer = None,
        segment_sentences: bool = False,
        max_sequence_length: int = None,
        skip_label_indexing: bool = False,
        text_key: str = "text",
        label_key: str = "label",
        task_name: str = "SST-5",
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._segment_sentences = segment_sentences
        self._max_sequence_length = max_sequence_length
        self._skip_label_indexing = skip_label_indexing
        self._token_indexers = token_indexers
        self._text_key = text_key
        self._label_key = label_key
        self.task_name = task_name
        if self._segment_sentences:
            self._sentence_segmenter = SpacySentenceSplitter()

    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            for line in self.shard_iterable(data_file.readlines()):
                # SST-5和其它数据集的格式不一样
                if self.task_name == "SST-5":
                    # SST-5 使用以下代码
                    if not line and len(len) < 3:
                        continue
                    line = line.strip()
                    text, label = line[2:], line[0]
                else:
                    # CR, MPQA, MR, SST-2, SUBJ 使用以下代码
                    if len(line.strip().split('\t')) != 2:
                        continue
                    text, label = line.strip().split("\t")
                
                tokens = self._tokenizer.tokenize(text)

                if self._max_sequence_length:
                    tokens = tokens[: self._max_sequence_length]
                text_field = TextField(tokens, self._token_indexers)
                label_field = LabelField(label)
                yield Instance({"tokens": text_field, "label": label_field})
                
    def _truncate(self, tokens):
        """
        truncate a set of tokens using the provided sequence length 使用提供的序列长度截断一组令牌
        """
        if len(tokens) > self._max_sequence_length:
            tokens = tokens[: self._max_sequence_length]
        return tokens
