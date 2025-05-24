from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F, Linear
from tacotron2.layers import ConvNorm, LinearNorm
from tacotron2.utils import to_gpu, get_mask_from_lengths


class LocationLayer(nn.Module):
    def __init__(self, n_filters, kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        self.location_conv = nn.Conv1d(2, n_filters, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                       bias=False)
        self.location_dense = nn.Linear(n_filters, attention_dim, bias=False)

    def forward(self, attention_weights_cat):
        processed = self.location_conv(attention_weights_cat)
        processed = processed.transpose(1, 2)
        processed = self.location_dense(processed)
        return processed


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))
        energy_bias = torch.linspace(-1.0, 1.0, energies.size(1), device=energies.device)
        energies = energies + energy_bias.unsqueeze(0)  # [1, T]

        energies = energies.squeeze(-1)

        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment = alignment.masked_fill(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.hparams = hparams
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = nn.Sequential(
            nn.Linear(hparams.n_mel_channels, hparams.prenet_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hparams.prenet_dim, hparams.prenet_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        input_dim = hparams.prenet_dim + hparams.encoder_embedding_dim + hparams.emotion_embedding_dim
        self.attention_rnn = nn.LSTMCell(input_dim, hparams.attention_rnn_dim)

        self.location_attention = LocationSensitiveAttention(
            attention_rnn_dim=hparams.attention_rnn_dim,
            encoder_embedding_dim=hparams.encoder_embedding_dim,
            attention_dim=hparams.attention_dim,
            attention_location_n_filters=hparams.attention_location_n_filters,
            attention_location_kernel_size=hparams.attention_location_kernel_size
        )

        self.decoder_rnn = nn.LSTMCell(hparams.attention_rnn_dim + hparams.encoder_embedding_dim,
                                       hparams.decoder_rnn_dim)

        self.linear_projection = nn.Linear(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim, hparams.n_mel_channels)
        self.gate_layer = nn.Linear(hparams.decoder_rnn_dim + hparams.encoder_embedding_dim, 1)

        self.memory_layer = nn.Linear(hparams.encoder_embedding_dim, hparams.attention_dim)  # Outputs attention_dim

    def get_go_frame(self, memory):
        return memory.new_full((memory.size(0), self.n_mel_channels), fill_value=-5.0) + 0.5

    def initialize_decoder_states(self, memory, mask):
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = memory.new_zeros(B, self.attention_rnn.hidden_size)
        self.attention_cell = memory.new_zeros(B, self.attention_rnn.hidden_size)
        self.decoder_hidden = memory.new_zeros(B, self.decoder_rnn.hidden_size)
        self.decoder_cell = memory.new_zeros(B, self.decoder_rnn.hidden_size)

        self.attention_weights = memory.new_zeros(B, MAX_TIME)
        self.attention_weights[:, 0] = 1.0  # Начинаем с первого токена
        self.attention_weights[:, :3] = torch.tensor([0.5, 0.3, 0.2], device=memory.device).unsqueeze(0)
        self.attention_context = memory.new_zeros(memory.size(0), memory.size(2))  # [B, encoder_embedding_dim]

        self.cumulative_attention_weights = memory.new_zeros(B, MAX_TIME)
        self.attention_weights = memory.new_zeros(B, MAX_TIME)
        if MAX_TIME > 1:
            start = 0
            end = min(5, MAX_TIME - 1)
            self.attention_weights[:, start:end] = 1.0 / (end - start)
        else:
            self.attention_weights[:, 0] = 1.0

        self.processed_memory = self.memory_layer(memory)  # [B, T, 128] → [B, T, 256]

    def parse_decoder_inputs(self, decoder_inputs):
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        alignments = torch.stack(alignments).transpose(0, 1)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1).contiguous()
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        mel_outputs = mel_outputs.transpose(1, 2)
        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input, memory, processed_memory, emotion_embedded, mask, step=None, epoch: int = 0):
        coverage_loss = 0.0
        if emotion_embedded.dim() == 3 and emotion_embedded.size(1) == 1:
            emotion_embedded = emotion_embedded.squeeze(1)
        elif emotion_embedded.dim() == 1:
            emotion_embedded = emotion_embedded.unsqueeze(0)  # [1, 64]  # [B, 64] вместо [B, 1, 64]
        cell_input = torch.cat((decoder_input, self.attention_context, emotion_embedded), dim=-1)

        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell)
        )

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1), self.cumulative_attention_weights.unsqueeze(1)), dim=1
        )

        attention_context, attention_weights = self.location_attention(
            self.attention_hidden, memory, self.processed_memory,
            attention_weights_cat, mask, step, epoch,
            max_decoder_steps=self.max_decoder_steps
        )

        B, T_enc = attention_weights.size()
        position = torch.arange(T_enc, device=attention_weights.device).unsqueeze(0).expand(B, -1)

        # --- Сужаем окно внимания по ходу обучения ---
        if self.training and 3 <= epoch <= 50:
            current_idx = torch.argmax(attention_weights, dim=1)
            window_size = min(max(20, 80 - epoch // 2), 40)
            start = (current_idx - window_size).clamp(min=0).unsqueeze(1)
            end = (current_idx + window_size).clamp(max=T_enc - 1).unsqueeze(1)

            mask_window = (position >= start) & (position <= end)
            outside_factor = max(0.01, 0.05 - epoch * 0.0005)
            attention_weights = torch.where(mask_window, attention_weights, attention_weights * outside_factor)

        # --- Запрещаем прыжки назад после 5-й эпохи ---
        if self.training and 5 <= epoch < 55:
            prev_pos = torch.argmax(self.attention_weights, dim=1)
            curr_pos = torch.argmax(attention_weights, dim=1)
            rollback = (curr_pos + 5 < prev_pos).unsqueeze(1)
            attention_weights = torch.where(rollback, torch.zeros_like(attention_weights), attention_weights)
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)

        # --- Обновляем состояние внимания ---
        self.attention_weights = attention_weights
        self.cumulative_attention_weights += attention_weights

        coverage = self.cumulative_attention_weights
        coverage = coverage / (coverage.sum(dim=1, keepdim=True) + 1e-8)
        coverage_penalty = torch.clamp(1.0 - coverage, min=0.0)
        self.attention_context = attention_context

        decoder_input = torch.cat((self.attention_hidden, self.attention_context), dim=-1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell)
        )

        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        if epoch >= 40:
            final_coverage = self.cumulative_attention_weights[:, -1]
            penalty = torch.clamp(1.0 - final_coverage, min=0.0)
            coverage_loss += penalty.mean() * 0.05

        return decoder_output, gate_prediction, attention_weights, coverage_loss

    def forward(self, memory, decoder_inputs, memory_lengths, emotion_embedded=None, epoch: int = 0):

        mask = ~get_mask_from_lengths(memory_lengths)
        self.initialize_decoder_states(memory, mask)

        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        T = decoder_inputs.size(0)

        if emotion_embedded is not None:
            if emotion_embedded.dim() == 2:
                emotion_embedded = emotion_embedded.unsqueeze(1).expand(-1, T, -1)
            elif emotion_embedded.dim() == 3 and emotion_embedded.size(1) != T:
                if emotion_embedded.size(1) > T:
                    emotion_embedded = emotion_embedded[:, :T, :]
                else:
                    pad_len = T - emotion_embedded.size(1)
                    pad = emotion_embedded[:, -1:, :].expand(-1, pad_len, -1)
                    emotion_embedded = torch.cat([emotion_embedded, pad], dim=1)
            emotion_embedded = emotion_embedded.transpose(0, 1)

        mel_outputs, gate_outputs, alignments = [], [], []

        tf_ratio = max(self.hparams.teacher_forcing_final,
                       self.hparams.teacher_forcing_start - epoch / self.hparams.teacher_forcing_decay_epochs)

        early_stop = torch.zeros(memory.size(0), dtype=torch.bool, device=memory.device)

        for t in range(T):
            tf_step_ratio = tf_ratio * max(0.5, 1.0 - t / T)
            if t == 0 or torch.rand(1).item() < tf_step_ratio or mel_output is None:
                decoder_input = self.prenet(decoder_inputs[t])
            else:
                decoder_input = self.prenet(mel_output.detach())
            emo_t = emotion_embedded[t] if emotion_embedded is not None else torch.zeros(
                memory.size(0), self.hparams.emotion_embedding_dim).to(memory.device)

            mel_output, gate_output, alignment, coverage_loss = self.decode(
                decoder_input,
                memory,
                self.processed_memory,
                emotion_embedded=emo_t,
                mask=mask,
                step=t,
                epoch=epoch
            )
            gate_sigmoid = torch.sigmoid(gate_output).squeeze(1)
            early_stop = early_stop | (gate_sigmoid > self.gate_threshold)

            if not self.training and early_stop.all():
                break

            mel_outputs.append(mel_output.unsqueeze(0))
            gate_outputs.append(gate_output.unsqueeze(0))
            alignments.append(alignment.unsqueeze(0))

        mel_outputs = torch.cat(mel_outputs, dim=0).transpose(0, 1)
        gate_outputs = torch.cat(gate_outputs, dim=0).transpose(0, 1).squeeze(-1)
        alignments = torch.cat(alignments, dim=0).transpose(0, 1)

        return mel_outputs, gate_outputs, alignments, coverage_loss

    def inference(self, memory, emotion_embedding=None):
        device = memory.device
        decoder_input = self.get_go_frame(memory)  # [B, n_mel_channels]
        decoder_input = decoder_input + 0.1
        memory_lengths = torch.full((memory.size(0),), memory.size(1), dtype=torch.long, device=device)
        mask = ~get_mask_from_lengths(memory_lengths)

        self.initialize_decoder_states(memory, mask)
        self.processed_memory = self.memory_layer(memory)

        if emotion_embedding is None:
            emotion_embedding = torch.zeros(memory.size(0), self.hparams.emotion_embedding_dim).to(device)

        mel_outputs, gate_outputs, alignments = [], [], []
        step = 0
        min_decoder_steps = 80

        while True:
            decoder_input_prenet = self.prenet(decoder_input)

            emo_step = emotion_embedding

            mel_output, gate_output, alignment, _ = self.decode(
                decoder_input_prenet, memory, self.processed_memory, emo_step, mask, step=step, epoch=20
            )

            mel_outputs.append(mel_output.unsqueeze(0))
            gate_outputs.append(gate_output.unsqueeze(0))
            alignments.append(alignment.unsqueeze(0))

            if step > min_decoder_steps and torch.sigmoid(gate_output).item() > self.gate_threshold:
                break
            if step >= self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output
            step += 1

        mel_outputs = torch.cat(mel_outputs, dim=0).transpose(0, 1)
        gate_outputs = torch.cat(gate_outputs, dim=0).transpose(0, 1).squeeze(-1)
        alignments = torch.cat(alignments, dim=0).transpose(0, 1)

        return mel_outputs, gate_outputs, alignments


class LocationSensitiveAttention(nn.Module):
    def __init__(self, attention_rnn_dim, encoder_embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super().__init__()
        self.query_layer = Linear(attention_rnn_dim, attention_dim, bias=False)
        self.memory_layer = Linear(encoder_embedding_dim, attention_dim, bias=False)
        self.v = Linear(attention_dim, 1, bias=True)
        self.location_layer = LocationLayer(attention_location_n_filters, attention_location_kernel_size, attention_dim)

        torch.nn.init.xavier_uniform_(self.query_layer.weight)
        torch.nn.init.xavier_uniform_(self.memory_layer.weight)
        torch.nn.init.xavier_uniform_(self.v.weight)
        torch.nn.init.constant_(self.v.bias, -1.0)
        torch.nn.init.xavier_uniform_(self.location_layer.location_dense.weight)

        self.score_mask_value = -float("inf")

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask=None, step=None, epoch=0,
                max_decoder_steps=None):
        processed_query = self.query_layer(attention_hidden_state).unsqueeze(1)
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_memory)).squeeze(-1)

        if self.training and step is not None and epoch >= 38 and max_decoder_steps is not None:
            B, T_enc = energies.size()
            ratio = step / max_decoder_steps
            center_val = ratio * (T_enc - 1)
            position = torch.arange(T_enc, device=energies.device).float()
            sigma = 3.0
            guide = torch.exp(-(position - center_val) ** 2 / (2 * sigma ** 2))
            guide = guide / (guide.sum() + 1e-8)
            alpha = 0.15
            energies = (1 - alpha) * energies + alpha * guide

        if self.training and epoch >= 40 and step is not None and max_decoder_steps is not None:
            ratio = step / max_decoder_steps
            if ratio > 0.6:
                bias_strength = (ratio - 0.6) * 1.2
                right_bias = torch.linspace(0.0, -bias_strength, energies.size(1), device=energies.device)
                energies += right_bias.unsqueeze(0)

        if mask is not None:
            energies = energies.masked_fill(mask, -10.0)
            all_inf = torch.isinf(energies).all(dim=1)
            if all_inf.any():
                for b in range(energies.size(0)):
                    if all_inf[b]:
                        energies[b, 0] = 0.0

        attention_weights = F.softmax(energies, dim=1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)

        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)
        return attention_context, attention_weights


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            print("memory_lengths:", output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
