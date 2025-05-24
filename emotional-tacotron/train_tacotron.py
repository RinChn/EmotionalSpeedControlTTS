import librosa
import re
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from num2words import num2words
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from tacotron2.hparams import create_hparams
from tacotron2.model import Encoder, Decoder, Postnet
from tacotron2.utils import to_gpu, get_mask_from_lengths
import torch.nn.functional as F

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

MODEL_SAVE_DIR = "model_and_alignment"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


class SpeakerEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(hparams.n_mel_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(256, hparams.speaker_embedding_dim // 2, bidirectional=True)
        self.proj = nn.Linear(hparams.speaker_embedding_dim, hparams.speaker_embedding_dim)

    def forward(self, mel_spectrogram):
        x = self.conv_layers(mel_spectrogram)
        x = x.permute(2, 0, 1)  # [T, B, C]
        _, (h, _) = self.lstm(x)
        h = h.permute(1, 0, 2).reshape(h.size(1), -1)
        return self.proj(h)


class SpeechMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mel_losses = []
        self.gate_losses = []
        self.emotion_losses = []
        self.alignment_scores = []

    def calculate_alignment_score(self, alignments):
        """Вычисление оценки выравнивания (attention)"""
        if alignments is None:
            return 0.0

        alignments = alignments.float()
        batch_size, _, steps = alignments.size()
        score = 0.0
        for b in range(batch_size):
            for t in range(1, steps):
                pos_prev = torch.argmax(alignments[b, :, t - 1])
                pos_curr = torch.argmax(alignments[b, :, t])
                score += torch.abs(pos_curr - pos_prev).float()
        return (score / (batch_size * steps)).item()

    def update(self, mel_loss, gate_loss, emotion_loss, alignments=None):
        self.mel_losses.append(mel_loss)
        self.gate_losses.append(gate_loss)
        self.emotion_losses.append(emotion_loss)

        if alignments is not None:
            alignment_score = self.calculate_alignment_score(alignments)
            self.alignment_scores.append(alignment_score)

    def get_metrics(self):
        def safe_mean(values):
            if not values:
                return 0.0
            return float(sum(values) / len(values))

        return {
            'avg_mel_loss': safe_mean(self.mel_losses),
            'avg_gate_loss': safe_mean(self.gate_losses),
            'avg_emotion_loss': safe_mean(self.emotion_losses),
            'avg_alignment_score': safe_mean(self.alignment_scores)
        }


class EmotionDataset(Dataset):
    def __init__(self, csv_file, emotion_to_label, dataset_type="train"):
        try:
            self.data = pd.read_csv(csv_file).reset_index(drop=True)
            self.data = self.data.dropna(subset=['path', 'text', 'emotion'])
            self.data = self.data.reset_index(drop=True)

            self.emotion_to_label = emotion_to_label
            self.dataset_type = dataset_type
            self.base_path = os.path.dirname(csv_file)
            self.audio_base_path = os.path.join(self.base_path, dataset_type)

            chars = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя .,!?-:;…'\""
            self.char_to_index = {char: idx + 1 for idx, char in enumerate(chars)}
            self.char_to_index[''] = 0
            self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}

            self.mel_mean = -5.0
            self.mel_std = 2.0

            print(f"Успешно загружен датасет: {len(self.data)} примеров")
        except Exception as e:
            print(f"Ошибка при загрузке датасета: {str(e)}")
            self.data = pd.DataFrame()

    @staticmethod
    def clean_text(text):
        if not isinstance(text, str):
            return ""

        char_replacements = {
            '«': '"', '»': '"', '—': '-', 'ё': 'е', '…': '...'
        }

        for char, replacement in char_replacements.items():
            text = text.replace(char, replacement)

        def replace_numbers(match):
            num_str = match.group().replace(',', '.')
            try:
                if '.' in num_str:
                    return num2words(float(num_str), lang='ru')
                return num2words(int(num_str), lang='ru')
            except:
                return num_str

        text = re.sub(r'(\d+[,.]?\d*)', replace_numbers, text)

        allowed = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя .,!?-")
        return ''.join(c for c in text if c.lower() in allowed)

    def text_to_sequence(self, text):
        return [self.char_to_index.get(char.lower(), 0) for char in text]

    def __len__(self):
        return len(self.data)

    def _compute_stats(self):
        all_mels = []
        for idx in tqdm(range(len(self.data))):
            item = self.__getitem__(idx)
            if item is not None:
                all_mels.append(item['mel'])
        all_mels = np.concatenate(all_mels, axis=0)
        return np.mean(all_mels), np.std(all_mels)

    def __getitem__(self, idx):
        try:
            if idx >= len(self.data):
                return None

            row = self.data.iloc[idx]
            if 'path' not in row or 'text' not in row or 'emotion' not in row:
                return None

            relative_path = row['path']
            full_path = os.path.join(self.audio_base_path, relative_path).replace("\\", "/")

            if not os.path.exists(full_path):
                return None

            audio_data, _ = librosa.load(full_path, sr=22050)
            audio_data = librosa.util.normalize(audio_data) * 0.9

            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio_data, sr=22050, n_mels=80,
                n_fft=1024, hop_length=256, win_length=1024,
                fmin=0, fmax=11025
            )

            mel_spectrogram = np.clip(mel_spectrogram, 1e-10, None)
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=1.0)
            log_mel_spectrogram = (log_mel_spectrogram - self.mel_mean) / self.mel_std

            raw_text = row['text']
            text = self.clean_text(raw_text)
            emotion = row['emotion']

            if len(text) == 0 or mel_spectrogram.shape[1] == 0:
                return None

            return {
                'text': self.text_to_sequence(text),
                'text_length': len(text),
                'mel': log_mel_spectrogram.T,
                'mel_length': log_mel_spectrogram.shape[1],
                'emotion_label': self.emotion_to_label[emotion]
            }
        except Exception as e:
            print(f"Ошибка при обработке элемента {idx}: {str(e)}")
            return None


def collate_fn(batch):
    batch = [item for item in batch if item is not None
             and item.get('text') and item.get('mel') is not None
             and len(item['text']) > 0 and item['mel'].size > 0 and not np.isnan(item['mel']).any()]

    if not batch:
        return None

    batch.sort(key=lambda x: -x['text_length'])

    texts = [torch.tensor(item['text'], dtype=torch.long) for item in batch]
    text_lengths = torch.tensor([item['text_length'] for item in batch], dtype=torch.long)
    mels = [torch.tensor(item['mel'].T, dtype=torch.float32) for item in batch]  # [n_mels, T]
    mel_lengths = torch.tensor([item['mel_length'] for item in batch], dtype=torch.long)
    emotion_labels = torch.tensor([item['emotion_label'] for item in batch], dtype=torch.long)

    max_text_len = max(len(text) for text in texts)
    max_mel_len = max(mel.shape[1] for mel in mels)

    text_padded = torch.zeros(len(texts), max_text_len, dtype=torch.long)
    for i, text in enumerate(texts):
        text_padded[i, :len(text)] = text

    mel_padded = torch.zeros(len(mels), mels[0].shape[0], max_mel_len)  # [B, n_mels, T]
    for i, mel in enumerate(mels):
        mel_padded[i, :, :mel.shape[1]] = mel

    gate_padded = torch.zeros(len(mels), max_mel_len)
    for i, length in enumerate(mel_lengths):
        if length > 0:
            gate_padded[i, length - 1:] = 1.0

    return {
        'text': text_padded,
        'text_length': text_lengths,
        'mel': mel_padded,
        'mel_length': mel_lengths,
        'emotion_label': emotion_labels,
        'gate': gate_padded
    }


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.hparams = hparams

        self.embedding = nn.Embedding(
            hparams.n_symbols,
            hparams.symbols_embedding_dim
        )
        self.emotion_embedding = nn.Embedding(
            hparams.n_emotions,
            hparams.emotion_embedding_dim
        )

        self.speaker_encoder = SpeakerEncoder(hparams)
        self.projection = nn.Sequential(
            nn.Linear(
                hparams.symbols_embedding_dim +
                hparams.emotion_embedding_dim +
                hparams.speaker_embedding_dim,
                512
            ),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        for name, param in self.decoder.location_attention.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=1.0)
            if 'bias' in name:
                nn.init.constant_(param, 0.1)
        self.postnet = Postnet(hparams)

        self.emotion_classifier = nn.Sequential(
            nn.Linear(self.hparams.n_mel_channels + 1, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.hparams.n_emotions)
        )
        nn.init.xavier_uniform_(self.emotion_classifier[0].weight, gain=0.3)
        nn.init.xavier_uniform_(self.emotion_classifier[3].weight, gain=0.3)

    def parse_batch(self, batch):
        text_padded = batch['text']
        input_lengths = batch['text_length']
        mel_padded = batch['mel']  # [B, n_mels, T]
        output_lengths = batch['mel_length']
        emotion_labels = batch['emotion_label']
        gate_padded = batch['gate']

        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        emotion_labels = to_gpu(emotion_labels).long()

        return (
            (text_padded, input_lengths, mel_padded, output_lengths, emotion_labels),
            (mel_padded, gate_padded)
        )

    def forward(self, inputs, epoch: int = 0):
        text_inputs, input_lengths, mels, output_lengths, emotion_labels = inputs

        embedded_text = self.embedding(text_inputs)  # [B, T, 256]
        emotion_embedded = self.emotion_embedding(emotion_labels)  # [B, 64]
        speaker_embedded = self.speaker_encoder(mels)  # [B, 128]

        emotion_embedded = emotion_embedded.unsqueeze(1).expand(-1, embedded_text.size(1), -1)
        speaker_embedded = speaker_embedded.unsqueeze(1).expand(-1, embedded_text.size(1), -1)

        combined = torch.cat([embedded_text, emotion_embedded, speaker_embedded], dim=-1)

        projected = self.projection(combined)  # [B, T, 512]
        encoder_input = projected.transpose(1, 2)  # [B, 512, T]

        encoder_outputs = self.encoder(encoder_input, input_lengths)

        if mels.size(1) != self.hparams.n_mel_channels:
            mels = mels.transpose(1, 2)

        mel_outputs, gate_outputs, alignments, coverage_loss = self.decoder(
            encoder_outputs, mels, memory_lengths=input_lengths,
            emotion_embedded=emotion_embedded,
            epoch=epoch
        )

        mel_outputs_postnet = self.postnet(mel_outputs.permute(0, 2, 1))

        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2) + mel_outputs

        emotion_features = self._extract_emotion_features(mel_outputs_postnet, alignments)
        emotion_pred = self.emotion_classifier(emotion_features)

        return {
            'mel_outputs': mel_outputs,
            'mel_outputs_postnet': mel_outputs_postnet,
            'gate_outputs': gate_outputs,
            'alignments': alignments,
            'emotion_pred': emotion_pred,
            'coverage_loss': coverage_loss
        }

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)
            if len(outputs) > 3 and outputs[3] is not None:
                outputs[3].data.masked_fill_(mask[:, 0, :], 0.0)

        return outputs

    def inference(self, inputs, speaker_embeddings=None, emotion_embedding=None):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs, emotion_embedding=emotion_embedding
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return mel_outputs_postnet, gate_outputs, alignments

    def text_to_sequence(self, text):
        """Преобразование текста в последовательность индексов"""
        chars = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя .,!?-:;'\""
        char_to_index = {char: idx + 1 for idx, char in enumerate(chars)}
        char_to_index[''] = 0
        return [char_to_index.get(char.lower(), 0) for char in text]

    def synthesize(self, text, emotion_id):
        self.eval()
        device = next(self.parameters()).device

        # 1. Токенизируем текст
        sequence = torch.LongTensor([self.text_to_sequence(text)]).to(device)
        input_lengths = torch.LongTensor([sequence.size(1)]).to(device)

        # 2. Преобразуем текст в эмбеддинги
        embedded_text = self.embedding(sequence).transpose(1, 2)

        # 3. Speaker embedding
        dummy_mel = torch.randn(1, self.hparams.n_mel_channels, 80).to(device) * 0.5
        speaker_embedding = self.speaker_encoder(dummy_mel)

        # 4. Emotion embedding
        emotion = torch.LongTensor([emotion_id]).to(device)
        emotion_embedding_raw = self.emotion_embedding(emotion)

        # 5. Расширение speaker/emotion для encoder
        T = embedded_text.shape[-1]
        speaker_embedding_exp = speaker_embedding.unsqueeze(-1).expand(-1, -1, T)
        emotion_embedding_exp = emotion_embedding_raw.unsqueeze(-1).expand(-1, -1, T)

        encoder_input = torch.cat([embedded_text, speaker_embedding_exp, emotion_embedding_exp], dim=1)

        # 6. Encoder
        memory = self.encoder(encoder_input, input_lengths)

        # 7. Decoder
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            memory,
            emotion_embedding=emotion_embedding_raw  # передаём правильный (сырой) emotion
        )

        # 8. Postnet
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        # 9. Денормализация
        mel_outputs_postnet = mel_outputs_postnet * self.hparams.mel_std + self.hparams.mel_mean

        # 10. Перевод из dB в power
        mel_outputs_postnet = mel_outputs_postnet.squeeze(0).cpu().detach().numpy()

        # 11. В numpy
        mel_outputs_postnet = torch.from_numpy(mel_outputs_postnet).unsqueeze(0)
        mel_outputs_postnet = mel_outputs_postnet.to(device)
        mel_outputs_postnet = mel_outputs_postnet.squeeze(0).cpu().detach().numpy()

        return {
            "mel": mel_outputs_postnet,
            "alignments": alignments,
            "sample_rate": self.hparams.sampling_rate,
        }

    def get_emotion_embedding(self, emotion_ids):
        return self.emotion_embedding(emotion_ids)

    def mel_to_audio(self, mel, n_iter=100):
        """Преобразование MEL-спектрограммы в аудиосигнал"""
        hparams = self.hparams

        # Преобразование в линейную спектрограмму
        mel_basis = librosa.filters.mel(
            sr=hparams.sampling_rate,
            n_fft=hparams.filter_length,
            n_mels=hparams.n_mel_channels,
            fmin=hparams.mel_fmin,
            fmax=hparams.mel_fmax
        )
        inv_mel_basis = np.linalg.pinv(mel_basis)
        spec = np.dot(inv_mel_basis, librosa.db_to_power(mel))

        # Griffin-Lim алгоритм
        waveform = librosa.griffinlim(
            spec,
            n_iter=n_iter,
            hop_length=hparams.hop_length,
            win_length=hparams.win_length,
            n_fft=hparams.filter_length
        )

        # Нормализация и обрезка тишины
        waveform = librosa.effects.trim(waveform, top_db=25)[0]
        return librosa.util.normalize(waveform) * 0.9

    def mel_to_audio_enhanced(self, mel, n_iter=200):
        # Жесткая нормализация MEL
        mel = np.clip(mel, -20, 20) + 40  # Смещение +40dB

        # Преобразование в линейную спектрограмму
        mel_basis = librosa.filters.mel(
            sr=self.hparams.sampling_rate,
            n_fft=self.hparams.filter_length,
            n_mels=self.hparams.n_mel_channels,
            fmin=self.hparams.mel_fmin,
            fmax=self.hparams.mel_fmax
        )
        inv_mel_basis = np.linalg.pinv(mel_basis)
        spec = np.dot(inv_mel_basis, librosa.db_to_power(mel))

        # Улучшенный Griffin-Lim с фазовой инициализацией
        waveform = librosa.griffinlim(
            spec,
            n_iter=n_iter,
            hop_length=self.hparams.hop_length,
            win_length=self.hparams.win_length,
            n_fft=self.hparams.filter_length,
            momentum=0.99,
            init='random',
            random_state=42
        )

        # Постобработка аудио
        waveform = librosa.effects.trim(waveform, top_db=25)[0]
        return librosa.util.normalize(waveform) * 0.9

    def _extract_emotion_features(self, mels, alignments):
        """Альтернативный способ извлечения признаков для классификации эмоций"""
        # 1. Используем среднее значение по MEL-спектрограммам
        mel_features = torch.mean(mels, dim=1)

        # 2. Добавляем статистики по alignments
        align_mean = torch.mean(alignments, dim=2)  # [B, text_len]
        align_features = torch.mean(align_mean, dim=1, keepdim=True)  # [B, 1]

        # 3. Объединяем признаки
        features = torch.cat([mel_features, align_features], dim=1)

        return features


def load_checkpoint(model, optimizer, scaler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False,
                            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    epoch = checkpoint['epoch']
    return epoch


def guided_attention_loss(attn, input_lengths, output_lengths, epoch, g=0.4):
    B, T_out, T_in = attn.size()
    device = attn.device

    grid_i = torch.arange(T_out, device=device).float().unsqueeze(1) / T_out
    grid_j = torch.arange(T_in, device=device).float().unsqueeze(0) / T_in
    guided_mask = 1.0 - torch.exp(-((grid_i - grid_j) ** 2) / (2 * g * g))
    guided_mask = guided_mask.unsqueeze(0).expand(B, -1, -1)

    # Маскируем guided_mask на паддинги
    for b in range(B):
        guided_mask[b, output_lengths[b]:, :] = 0.0
        guided_mask[b, :, input_lengths[b]:] = 0.0

    loss = torch.mean(attn * guided_mask) * 2.0
    if epoch > 35:
        return loss * 2.0
    return loss * 8.0


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    emotion_to_label = {
        'neutral': 0, 'anger': 1, 'enthusiasm': 2, 'fear': 3,
        'sadness': 4, 'happiness': 5, 'disgust': 6
    }

    train_dataset = EmotionDataset("../dataset/train.csv", emotion_to_label, "train")
    train_dataset.original_data = train_dataset.data.copy()
    test_dataset = EmotionDataset("../dataset/test.csv", emotion_to_label, "test")

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Один из датасетов пуст!")

    print("Проверка примеров:")
    for i in range(3):
        sample = train_dataset[i]
        if sample:
            print(f"Пример {i}: текст длина - {sample['text_length']}, MEL длина - {sample['mel_length']}")

    durations = [librosa.get_duration(filename=os.path.join(train_dataset.audio_base_path, f))
                 for f in train_dataset.data['path'].sample(100)]
    print(f"Средняя длительность: {np.mean(durations):.2f} сек")

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True)

    hparams = create_hparams()
    hparams.n_emotions = len(emotion_to_label)

    model = Tacotron2(hparams).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=1e-6,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    num_epochs = 60
    total_training_steps = num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-6,
        total_steps=total_training_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        cycle_momentum=False,
        div_factor=1.0,
        final_div_factor=1.0
    )

    def compute_loss(outputs, targets, emotion_labels, input_lengths, output_lengths, epoch):
        mel_target, gate_target = targets
        mel_target = mel_target.transpose(1, 2)
        mel_out = outputs['mel_outputs']
        mel_target = mel_target

        min_len = min(mel_out.size(1), mel_target.size(1))
        mel_out = mel_out[:, :min_len, :]
        mel_target = mel_target[:, :min_len, :]

        # === Mel loss ===
        mel_loss = F.mse_loss(mel_out, mel_target, reduction='mean') * 0.5

        # === Gate loss ===
        gate_weight = 0.01 if epoch < 5 else 0.05
        min_gate_len = min(outputs['gate_outputs'].size(1), gate_target.size(1))
        gate_out = outputs['gate_outputs'][:, :min_gate_len]
        gate_target = gate_target[:, :min_gate_len]
        gate_loss = F.binary_cross_entropy_with_logits(gate_out, gate_target, reduction='mean') * gate_weight

        # === Emotion loss ===
        emotion_weight = max(0.05, 0.2 - 0.01 * epoch)
        emotion_loss = F.cross_entropy(outputs['emotion_pred'], emotion_labels, reduction='mean') * emotion_weight

        # === Alignment losses ===
        alignments = outputs['alignments']
        B, T_dec, T_enc = alignments.size()
        device = alignments.device

        # Guided attention loss+
        pos_dec = torch.arange(T_dec, device=device).float().unsqueeze(1) / (T_dec - 1)
        pos_enc = torch.arange(T_enc, device=device).float().unsqueeze(0) / (T_enc - 1)
        guided_mask = 1.0 - torch.exp(-((pos_dec - pos_enc) ** 2) / (2 * 0.4 ** 2))
        guided_mask = guided_mask.unsqueeze(0).expand(B, -1, -1)
        guided_loss = torch.mean(alignments * guided_mask)
        guided_weight = (
            0.6 if epoch <= 30 else
            0.2 if epoch < 35 else
            0.1 if epoch < 50 else
            0.05 if epoch < 60 else
            0.0
        )

        total_loss = guided_loss * guided_weight

        # Diagonal mask loss
        t_dec = torch.arange(T_dec, device=device).float().unsqueeze(0).expand(B, -1)
        ideal_pos = (t_dec / (output_lengths.unsqueeze(1) - 1)) * (input_lengths.unsqueeze(1) - 1)
        pos_enc = torch.arange(T_enc, device=device).float().unsqueeze(0).unsqueeze(0).expand(B, T_dec, T_enc)
        ideal_pos = ideal_pos.unsqueeze(2)
        sigma = 2.5 if epoch < 30 else 3.5 if epoch < 38 else 5.0
        diagonal_mask = torch.exp(-((pos_enc - ideal_pos) ** 2) / (2 * sigma ** 2))
        diagonal_mask = diagonal_mask / (diagonal_mask.sum(dim=2, keepdim=True) + 1e-8)
        alignments = alignments / (alignments.sum(dim=2, keepdim=True) + 1e-8)
        diagonal_loss = F.mse_loss(alignments, diagonal_mask) * 0.5
        diagonal_weight = 0.0
        total_loss += diagonal_loss * diagonal_weight

        # KL / entropy (sharpness)
        align_safe = torch.clamp(alignments, min=1e-8)
        sharpness = torch.sum((align_safe - align_safe.max(dim=2, keepdim=True)[0]) ** 2, dim=2)
        sharpness_loss = torch.mean(sharpness)
        sharpness_weight = 0.0
        total_loss += sharpness_weight * sharpness_loss

        # End penalty: если attention не доходит до конца
        if epoch >= 40:
            max_attn = torch.argmax(alignments.mean(dim=1), dim=1)  # [B]
            end_penalty = (input_lengths.float() - max_attn.float()) / input_lengths.float()
            penalty_weight = 0.15 if epoch < 55 else 0.25
            total_loss += torch.mean(end_penalty) * penalty_weight

        # Coverage penalty
        coverage_loss = outputs.get("coverage_loss", 0.0)
        if epoch < 35 and isinstance(coverage_loss, torch.Tensor):
            total_loss += coverage_loss * hparams.coverage_weight

        # Суммируем остальные потери
        total_loss += mel_loss + gate_loss + emotion_loss

        return total_loss, mel_loss.item(), gate_loss.item(), emotion_loss.item()

    best_test_loss = float('inf')
    train_metrics = SpeechMetrics()
    test_metrics = SpeechMetrics()
    grad_clip_thresh = 1.0
    scaler = torch.cuda.amp.GradScaler(enabled=hparams.fp16_run)

    checkpoint_path = "model_and_alignment/best_model.pth"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, scaler, "model_and_alignment/checkpoint_epoch_43.pth")
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_metrics.reset()

        max_mel_len = min(900, 200 + epoch * 50)
        train_loader.dataset.data = train_dataset.original_data[
            train_dataset.original_data.apply(
                lambda row: librosa.get_duration(
                    filename=os.path.join(train_dataset.audio_base_path, row['path'])) * 80 < max_mel_len,
                axis=1
            )
        ]
        train_iter = tqdm(train_loader,
                          desc=f'Train Epoch {epoch + 1}/{num_epochs}',
                          bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

        for batch in train_iter:
            if batch is None:
                continue

            inputs, targets = model.parse_batch(batch)
            optimizer.zero_grad()

            # Используем современный AMP
            with torch.amp.autocast("cuda", enabled=hparams.fp16_run):
                outputs = model(inputs, epoch=epoch)
                total_loss, mel_loss, gate_loss, emotion_loss = compute_loss(
                    outputs, targets, inputs[4], inputs[1], inputs[3], epoch)
            torch.autograd.set_detect_anomaly(True)
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
            scaler.step(optimizer)
            scaler.update()

            if epoch >= 15:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 2e-05

            scheduler.step()

            if epoch >= 45:
                for group in optimizer.param_groups:
                    group['lr'] = max(group['lr'] * 0.95, 1.5e-5)

            train_metrics.update(
                mel_loss=mel_loss,
                gate_loss=gate_loss,
                emotion_loss=emotion_loss,
                alignments=outputs.get('alignments')
            )

            train_iter.set_postfix({
                'mel': f"{train_metrics.get_metrics()['avg_mel_loss']:.4f}",
                'gate': f"{train_metrics.get_metrics()['avg_gate_loss']:.4f}",
                'align': f"{train_metrics.get_metrics()['avg_alignment_score']:.2f}"
            })

        align = outputs['alignments'][0].detach().cpu().numpy()
        plt.figure(figsize=(10, 5))
        plt.imshow(align, aspect='auto', origin='lower')
        plt.title(f'Epoch {epoch + 1} Alignment')
        plt.savefig(f'model_and_alignment/alignment_epoch_{epoch + 1}.png')
        plt.close()

        model.eval()
        test_metrics.reset()
        current_test_loss = 0.0

        with torch.no_grad():
            val_iter = tqdm(test_loader,
                            desc=f'Validation Epoch {epoch + 1}',
                            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
            for batch in val_iter:
                if batch is None:
                    continue
                inputs, targets = model.parse_batch(batch)
                with torch.cuda.amp.autocast(enabled=hparams.fp16_run):
                    outputs = model(inputs)
                    total_loss, mel_loss, gate_loss, emotion_loss = compute_loss(
                        outputs, targets, inputs[4], inputs[1], inputs[3], epoch)
                current_test_loss += total_loss.item()
                test_metrics.update(mel_loss, gate_loss, emotion_loss, outputs['alignments'])

        avg_test_loss = current_test_loss / len(test_loader)
        align_score = test_metrics.get_metrics()['avg_alignment_score']
        if epoch < 10:
            pass
        else:
            scheduler.step()

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'hparams': hparams
        }

        torch.save(checkpoint, os.path.join(MODEL_SAVE_DIR, f'checkpoint_epoch_{epoch + 1}.pth'))

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(checkpoint, os.path.join(MODEL_SAVE_DIR, 'best_model.pth'))
            print(f"\nNew best model saved with test loss: {best_test_loss:.4f}")

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train - Mel: {train_metrics.get_metrics()['avg_mel_loss']:.4f} | "
              f"Gate: {train_metrics.get_metrics()['avg_gate_loss']:.4f} | "
              f"Align: {train_metrics.get_metrics()['avg_alignment_score']:.2f}")
        print(f"Test  - Mel: {test_metrics.get_metrics()['avg_mel_loss']:.4f} | "
              f"Gate: {test_metrics.get_metrics()['avg_gate_loss']:.4f} | "
              f"Align: {test_metrics.get_metrics()['avg_alignment_score']:.2f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Align mean diff: {train_metrics.get_metrics()['avg_alignment_score']:.2f}")


if __name__ == "__main__":
    train()
