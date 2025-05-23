import os
import csv
import torch
import numpy as np
import torchaudio
from tqdm import tqdm
from tacotron2.layers import TacotronSTFT

# === Параметры аудио (должны совпадать с Tacotron2) ===
sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024
n_mel_channels = 80
mel_fmin = 80.0
mel_fmax = 11025.0

# === Пути ===
csv_paths = ["../dataset/train.csv", "../dataset/test.csv"]
audio_base_path = "../dataset"
mel_save_dir = "../dataset/mel"
os.makedirs(mel_save_dir, exist_ok=True)

# === STFT ===
stft = TacotronSTFT(
    filter_length=filter_length,
    hop_length=hop_length,
    win_length=win_length,
    sampling_rate=sampling_rate,
    mel_fmin=mel_fmin,
    mel_fmax=mel_fmax,
    n_mel_channels=n_mel_channels,
)

# === Обработка каждого .csv ===
for csv_path in csv_paths:
    subfolder = "train" if "train" in csv_path else "test"

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc=f"Processing {csv_path}"):
            name = row["name"]
            rel_path = row["path"].replace("\\", "/")
            wav_path = os.path.normpath(os.path.join(audio_base_path, subfolder, rel_path))
            mel_path = os.path.join(mel_save_dir, name + ".mel.npy")

            if not os.path.isfile(wav_path):
                print(f"✗ File not found: {wav_path}")
                continue

            try:
                # Загружаем аудио (всегда float32 в [-1, 1] от torchaudio)
                audio_tensor, sr = torchaudio.load(wav_path)

                # Преобразуем в моно
                if audio_tensor.ndim > 1 and audio_tensor.shape[0] > 1:
                    audio_tensor = audio_tensor.mean(dim=0, keepdim=True)

                # Ресемплируем
                if sr != sampling_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
                    audio_tensor = resampler(audio_tensor)

                # Нормализация в [-1, 1], если выбивается
                max_val = torch.max(torch.abs(audio_tensor))
                if max_val > 1.0:
                    audio_tensor = audio_tensor / max_val

                # Спектрограмма
                mel = stft.mel_spectrogram(audio_tensor)
                mel = mel.squeeze(0).cpu().numpy()

                # Сохраняем
                np.save(mel_path, mel)

            except Exception as e:
                print(f"✗ Failed {wav_path}: {e}")
