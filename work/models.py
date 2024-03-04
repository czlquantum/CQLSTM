# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  models.py
@Time    :  2023/05/19 
@Author  :  chu
@Contact :  chuzlu123@163.com
@Desc    :  None
"""
from typing import Optional, Tuple
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Linear, Conv1d
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn import Activation
from allennlp.nn.util import min_value_of_dtype
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexReLU
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
from .utils import mine_complex_max_pool2d, abs_fn
from .qlstm_cell import QLSTMCell

@Seq2VecEncoder.register("CQLSTM")
class ComplexQNNSeq2Vec(Seq2VecEncoder):

    def __init__(
            self,
            embedding_dim: int,
            output_dim: Optional[int] = None,
            device: str = 'cuda:0'
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._output_dim = output_dim
        self._activation = ComplexReLU()
        self.device = device
        # self.gru = nn.GRU(self._embedding_dim, self._embedding_dim, 2)
        projector_real = nn.init.uniform_(
            torch.nn.Parameter(torch.Tensor(self._output_dim, 1), requires_grad=False))
        projector_imag = nn.init.uniform_(
            torch.nn.Parameter(torch.Tensor(self._output_dim, 1), requires_grad=False))
        projector = projector_real + 1j * projector_imag
        self.projector = projector
        self.cf1 = ComplexLinear(self._embedding_dim, self._output_dim)
        self.cf2 = ComplexLinear(self._output_dim, self._output_dim)

        self.lstm = QLSTMCell(
            self._output_dim,
            self._output_dim,
            n_qubits=4,
            n_qlayers=1
        )
        self.lstm_real = QLSTMCell(
            self._output_dim,
            self._output_dim,
            n_qubits=4,
            n_qlayers=1
        )
        self.lstm_imag = QLSTMCell(
            self._output_dim,
            self._output_dim,
            n_qubits=4,
            n_qlayers=1
        )
        self.fc = nn.Linear(self._output_dim,1)

        if output_dim:
            self.projection_layer = ComplexLinear(self._embedding_dim, output_dim)
            self._output_dim = output_dim
        else:
            self.projection_layer = None
            self._output_dim = self._embedding_dim

    def get_input_dim(self) -> int:
        return self._embedding_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    #
    def simple_measure(self, x: torch.Tensor):
        # x [batch, seq_len, emb_dim]
        pooler = torch.mean(x, dim=-2)  # [batch, emb_dim]
        return pooler.real

    def opt_measure(self, x: torch.Tensor):
        # x: [batch, seq_len, emb_dim]
        seq_len = x.shape[1]
        projector_real = nn.init.uniform_(torch.nn.Parameter(torch.Tensor(seq_len, self._output_dim), requires_grad=False)).to(x.device)
        projector_imag = nn.init.uniform_(torch.nn.Parameter(torch.Tensor(seq_len, self._output_dim), requires_grad=False)).to(x.device)
        projector = projector_real + 1j * projector_imag
        # print(projector.shape)
        # print(x.shape)
        prob = x.permute(0, 2, 1) @ projector  # [seq_len, 1] @ [[batch, emb_dim, seq_len]] = [batch, seq_len, 1]
        prob = prob.squeeze()
        return prob.abs()

    def project_measure(self, V: torch.Tensor, P: torch.Tensor = None):
        if P == None:
            P = self.projector @ self.projector.permute(1, 0)
        result = torch.cat([
            torch.diag(P @ batch).abs().unsqueeze(0)
            for batch in V
        ]).to(P.device)
        return result

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor):
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1)
        else:
            # If mask doesn't exist create one of shape (batch_size, num_tokens)
            mask = torch.ones(tokens.shape[0], tokens.shape[1], device=tokens.device).bool()
        # tokens = tokens.unsqueeze(1).type(torch.complex64)
        if tokens.dtype != torch.complex64:
            imag_tokens = torch.randn_like(tokens)
            tokens = tokens + 1j * imag_tokens
        
            ## tokens: {shape=[batch, seq_len, emb_dim], dtype=torch.complex64}
        result = self.cf1(tokens)  # x: {tokens=[batch, seq_len, output_dim], dtype=torch.complex64}
        result_real = self.lstm_real(result.real)
        result_imag = self.lstm_imag(result.imag)
        result = result_real + result_imag * 1j
        result = self.opt_measure(result)
        result = self.fc(result)
        result = result.squeeze()

        return result