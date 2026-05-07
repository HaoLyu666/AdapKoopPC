from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn


class DSEncoder(nn.Module):
    """Driving-style encoder used by AdapKoopnet.

    Layer attribute names intentionally match the original implementation so
    legacy state_dict checkpoints load without conversion.
    """

    def __init__(self, args: Mapping):
        super().__init__()
        self.device = args["device"]
        self.lstm_encoder_size = args["lstm_encoder_size"]
        self.n_head = args["n_head"]
        self.att_out = args["att_out"]
        self.in_length = args["in_length"]
        self.out_length = args["out_length"]
        self.f_length = args["f_length"]
        self.relu_param = args["relu"]
        self.traj_linear_hidden = args["lstm_encoder_size"]
        self.train_flag = args["train_flag"]
        self.use_elu = args["use_elu"]
        self.dropout = args["dropout"]

        self.linear1 = nn.Linear(self.f_length, self.traj_linear_hidden)
        self.lstm = nn.LSTM(self.traj_linear_hidden, self.lstm_encoder_size)
        self.activation = nn.ELU() if self.use_elu else nn.LeakyReLU(self.relu_param)

        self.qt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.kt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.vt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.project0 = nn.Linear(self.n_head * self.att_out, self.lstm_encoder_size)
        self.project1 = nn.Linear(self.lstm_encoder_size, self.lstm_encoder_size * 4)
        self.project2 = nn.Linear(self.lstm_encoder_size * 4, self.lstm_encoder_size)
        self.qtt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.ktt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.vtt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.project10 = nn.Linear(self.n_head * self.att_out, self.lstm_encoder_size)
        self.project11 = nn.Linear(self.lstm_encoder_size, self.lstm_encoder_size * 4)

        self.glu = GLU(
            input_size=self.lstm_encoder_size * 4,
            hidden_layer_size=self.lstm_encoder_size,
            dropout_rate=self.dropout,
        )
        self.Dropout = nn.Dropout(p=self.dropout)
        self.addAndNorm = AddAndNorm(self.lstm_encoder_size)
        self.style_pred = nn.Linear(self.lstm_encoder_size, 3)
        self.mapping = nn.Parameter(torch.Tensor(3, self.lstm_encoder_size))
        nn.init.xavier_uniform_(self.mapping, gain=1.414)

        self.d_model = self.traj_linear_hidden
        self.max_seq_len = self.in_length + 1
        self.register_buffer(
            "positional_encoding",
            self.generate_positional_encoding(),
            persistent=False,
        )
        self.register_buffer(
            "ds_aw",
            torch.zeros([3, self.in_length], dtype=torch.float32),
            persistent=False,
        )

    def generate_positional_encoding(self) -> torch.Tensor:
        pe = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        history_sequence = history.permute(1, 0, 2)
        if self.train_flag:
            history_windows = torch.zeros(
                [self.in_length, (self.out_length + 1) * history_sequence.shape[1], self.f_length],
                dtype=history_sequence.dtype,
                device=history_sequence.device,
            )
            for i in range(self.out_length + 1):
                history_windows[:, i * history_sequence.shape[1] : (i + 1) * history_sequence.shape[1], :] = history_sequence[
                    i : self.in_length + i, :, :
                ]
        else:
            history_windows = history_sequence

        history_embedding = self.activation(self.linear1(history_windows))
        positional_encoding = self.positional_encoding[:, : self.in_length, :].to(history_embedding.device)
        history_embedding = history_embedding + positional_encoding.permute(1, 0, 2).repeat(
            1,
            history_embedding.shape[1],
            1,
        )

        history_context = history_embedding.permute(1, 0, 2)

        query = torch.cat(
            torch.split(self.qt(history_context), int(history_context.shape[-1] / self.n_head), -1),
            0,
        )
        key = torch.cat(
            torch.split(self.kt(history_context), int(history_context.shape[-1] / self.n_head), -1),
            0,
        ).permute(0, 2, 1)
        value = torch.cat(
            torch.split(self.vt(history_context), int(history_context.shape[-1] / self.n_head), -1),
            0,
        )
        attention = torch.matmul(query, key) / math.sqrt(self.att_out)
        attention = torch.softmax(attention, -1)
        attended_history = torch.matmul(attention, value)
        attended_history = torch.cat(torch.split(attended_history, int(history_embedding.shape[1]), 0), -1)
        attended_history = self.activation(self.project0(attended_history))
        attended_history = self.addAndNorm(history_context, attended_history)
        feedforward_history = self.Dropout(self.activation(self.project1(attended_history)))
        history_features = self.Dropout(self.activation(self.project2(feedforward_history)))
        history_features = self.addAndNorm(attended_history, history_features)

        driving_style_token = positional_encoding[:, -1:, :].repeat(history_features.shape[0], 1, 1)
        style_query_sequence = torch.cat((history_features, driving_style_token), 1)

        style_query = torch.cat(
            torch.split(self.qtt(style_query_sequence), int(history_context.shape[-1] / self.n_head), -1),
            0,
        )
        style_key = torch.cat(
            torch.split(self.ktt(history_features), int(history_context.shape[-1] / self.n_head), -1),
            0,
        ).permute(0, 2, 1)
        style_value = torch.cat(
            torch.split(self.vtt(history_features), int(history_context.shape[-1] / self.n_head), -1),
            0,
        )
        style_attention = torch.matmul(style_query, style_key) / math.sqrt(self.att_out)
        style_attention = torch.softmax(style_attention, -1)
        style_context = torch.matmul(style_attention, style_value)
        style_context = torch.cat(torch.split(style_context, int(history_embedding.shape[1]), 0), -1)
        style_context = self.activation(self.project10(style_context))
        style_context = self.addAndNorm(style_query_sequence, style_context)
        style_feedforward = self.Dropout(self.activation(self.project11(style_context)))

        gated_style_context, _ = self.glu(style_feedforward)
        style_features = self.addAndNorm(style_context, gated_style_context)

        style_probability = F.softmax(self.style_pred(style_features[:, -1, :]), dim=-1).unsqueeze(1)
        hard_style_indicator = torch.zeros_like(style_probability, device=history_sequence.device)
        style_index = torch.argmax(style_probability.squeeze(), dim=-1)
        for sample_index in range(hard_style_indicator.shape[0]):
            hard_style_indicator[sample_index, :, style_index[sample_index]] = 1

        style_attention_weight = F.softmax(
            torch.matmul(hard_style_indicator, torch.matmul(self.mapping, style_features[:, :-1, :].permute(0, 2, 1))),
            dim=-1,
        )
        self.ds_aw.zero_()
        for style_id in range(3):
            selected = style_attention_weight.squeeze()[style_index == style_id]
            if selected.numel() > 0:
                self.ds_aw[style_id, :] = torch.sum(selected, dim=0)

        driving_style = torch.matmul(style_attention_weight, style_features[:, :-1, :]).permute(1, 0, 2)
        if self.train_flag:
            return torch.cat(torch.split(driving_style, history_sequence.shape[1], 1), 0)
        return driving_style


class StateEncoder(nn.Module):
    def __init__(self, args: Mapping):
        super().__init__()
        self.device = args["device"]
        self.lstm_encoder_size = args["lstm_encoder_size"]
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.train_flag = args["train_flag"]
        self.in_length = args["in_length"]
        self.f_length = args["f_length"]
        self.relu_param = args["relu"]
        self.traj_linear_hidden = args["traj_linear_hidden"]
        self.use_elu = args["use_elu"]
        self.dropout = args["dropout"]

        self.linear1 = nn.Linear(3, self.traj_linear_hidden)
        self.Dropout = nn.Dropout(p=self.dropout)
        self.addAndNorm = AddAndNorm(self.lstm_encoder_size)
        self.activation = nn.ELU() if self.use_elu else nn.LeakyReLU(self.relu_param)
        self.project0 = nn.Linear(self.lstm_encoder_size, self.lstm_encoder_size * 2)
        self.project1 = nn.Linear(self.lstm_encoder_size * 2, self.lstm_encoder_size)
        self.project2 = nn.Linear(self.lstm_encoder_size, self.lstm_encoder_size * 2)
        self.glu = GLU(
            input_size=self.lstm_encoder_size * 2,
            hidden_layer_size=self.lstm_encoder_size,
            dropout_rate=self.dropout,
        )

    def forward(self, Hist: torch.Tensor) -> torch.Tensor:
        Hist = Hist.permute(1, 0, 2)
        Hist = Hist[self.in_length - 1 :, :, :]
        states0 = self.activation(self.linear1(Hist))
        states1 = self.Dropout(self.tanh(self.project0(states0)))
        states2 = self.Dropout(self.tanh(self.project1(states1)))
        states2 = self.addAndNorm(states0, states2)
        states3 = self.Dropout(self.activation(self.project2(states2)))
        states, _ = self.glu(states3)
        return states


class AddAndNorm(nn.Module):
    def __init__(self, hidden_layer_size: int):
        super().__init__()
        self.normalize = nn.LayerNorm(hidden_layer_size)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor | None = None) -> torch.Tensor:
        if x3 is not None:
            x = torch.add(torch.add(x1, x2), x3)
        else:
            x = torch.add(x1, x2)
        return self.normalize(x)


class Encoder(nn.Module):
    def __init__(self, args: Mapping):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.device = args["device"]
        self.lstm_encoder_size = args["lstm_encoder_size"]
        self.dropout = args["dropout"]
        self.glu = GLU(
            input_size=self.lstm_encoder_size * 2,
            hidden_layer_size=self.lstm_encoder_size,
            dropout_rate=self.dropout,
        )

    def forward(self, state: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
        state_and_style = torch.cat((state, ds), dim=-1)
        koopman_state, _ = self.glu(state_and_style)
        return koopman_state


class KoopmanSpace(nn.Module):
    def __init__(self, args: Mapping):
        super().__init__()
        self.args = args
        self.device = args["device"]
        self.lstm_encoder_size = args["lstm_encoder_size"]
        self.train_flag = args["train_flag"]
        self.A = nn.Linear(self.lstm_encoder_size, self.lstm_encoder_size, bias=False)
        self.B = nn.Linear(1, self.lstm_encoder_size, bias=False)
        self.C = nn.Linear(self.lstm_encoder_size, 3, bias=False)
        self.dec = Decoder(self.args)
        self.linear_decoder = bool(args["liner_dec"])
        self.liner_dec = self.linear_decoder

    def forward(self, KS_ALL: torch.Tensor, next_vf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        next_vf = next_vf.unsqueeze(-1).permute(1, 0, 2)
        if self.train_flag:
            ks = KS_ALL[0:1, :, :]
        else:
            ks = KS_ALL
        KS_pre = ks
        for i in range(next_vf.shape[0]):
            ks = self.A(ks) + self.B(next_vf[i : i + 1, :, :])
            KS_pre = torch.cat((KS_pre, ks), dim=0)
        OS = self.C(KS_pre) if self.linear_decoder else self.dec(KS_pre)
        return KS_pre, OS


class GLU(nn.Module):
    def __init__(self, input_size: int, hidden_layer_size: int, dropout_rate: float | None = None):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.activation_layer = nn.Linear(input_size, hidden_layer_size)
        self.gated_layer = nn.Linear(input_size, hidden_layer_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dropout_rate is not None:
            x = self.dropout(x)
        activation = self.activation_layer(x)
        gated = self.sigmoid(self.gated_layer(x))
        return torch.mul(activation, gated), gated


class Decoder(nn.Module):
    def __init__(self, args: Mapping):
        super().__init__()
        self.device = args["device"]
        self.lstm_encoder_size = args["lstm_encoder_size"]
        self.tanh = nn.Tanh()
        self.relu_param = args["relu"]
        self.use_elu = args["use_elu"]
        self.dropout = args["dropout"]
        self.Dropout = nn.Dropout(p=self.dropout)
        self.addAndNorm = AddAndNorm(self.lstm_encoder_size)
        self.activation = nn.ELU() if self.use_elu else nn.LeakyReLU(self.relu_param)
        self.project0 = nn.Linear(self.lstm_encoder_size, self.lstm_encoder_size * 2)
        self.project1 = nn.Linear(self.lstm_encoder_size * 2, self.lstm_encoder_size)
        self.project2 = nn.Linear(self.lstm_encoder_size, self.lstm_encoder_size)
        self.project3 = nn.Linear(self.lstm_encoder_size, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states1 = self.Dropout(self.tanh(self.project0(x)))
        states2 = self.Dropout(self.tanh(self.project1(states1)))
        states2 = self.addAndNorm(x, states2)
        states3 = self.Dropout(self.activation(self.project2(states2)))
        return self.project3(states3)


KoopSpace = KoopmanSpace
Koop_space = KoopmanSpace
decoder = Decoder
