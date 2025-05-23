import os
import sys
import subprocess
import soundfile as sf
import librosa


def check_duration(file_path, min_minutes=10):
    y, sr = librosa.load(file_path, sr=None)
    duration_sec = len(y) / sr
    return duration_sec >= min_minutes * 60, duration_sec


def convert_mp3_to_wav(mp3_path, wav_path):
    command = ["ffmpeg", "-y", "-i", mp3_path, wav_path]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        print("Ошибка конвертации:", result.stderr.decode())
        sys.exit(1)


def resample_wav(input_path, output_path):
    # Приведение к 44.1 кГц и моно
    command = ["ffmpeg", "-y", "-i", input_path, "-ar", "44100", "-ac", "1", output_path]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        print("Ошибка ресемплирования:", result.stderr.decode())
        sys.exit(1)


def run_svc_preprocessing(speaker_id):
    # Подготовка структуры проекта
    subprocess.run([
        "svc", "pre-resample",
        "-i", "so-vits-svc-fork/dataset_raw",
        "-o", "so-vits-svc-fork/dataset"
    ], check=True)

    # Создание конфигурации
    subprocess.run(["svc", "pre-config", "--speaker", speaker_id], check=True)

    # Hubert-эмбеддинги
    subprocess.run(["svc", "pre-hubert"], check=True)


def train_model():
    subprocess.run(["svc", "train"], check=True)


def main(mp3_path, speaker_name):
    temp_raw = "temp_original.wav"
    temp_clean = "temp_ref_voice.wav"

    print("📥 Конвертация mp3 → wav...")
    convert_mp3_to_wav(mp3_path, temp_raw)

    print("⏱ Проверка длительности...")
    ok, duration = check_duration(temp_raw)
    if not ok:
        print(f"Ошибка: длительность {duration / 60:.2f} мин < 10 мин")
        sys.exit(2)

    print("🔁 Ресемплирование до 44.1 кГц, моно...")
    resample_wav(temp_raw, temp_clean)

    # Подготовка индивидуальной директории
    speaker_id = speaker_name.replace(" ", "_")
    ref_dir = os.path.join("so-vits-svc-fork", "dataset_raw", speaker_id)
    os.makedirs(ref_dir, exist_ok=True)
    os.replace(temp_clean, os.path.join(ref_dir, "ref_voice.wav"))

    print("⚙️ Запуск этапов svc: pre-resample, pre-config, pre-hubert...")
    run_svc_preprocessing(speaker_id)

    print("🎤 Обучение модели...")
    train_model()

    print("✅ Обучение завершено. Модель сохранена в папке logs/")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Использование: python train_new_voice.py path_to_mp3 \"ФИО преподавателя\"")
        sys.exit(1)

    mp3_path = sys.argv[1]
    speaker_name = sys.argv[2]
    main(mp3_path, speaker_name)
