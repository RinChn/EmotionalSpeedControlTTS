import glob
import os
import subprocess
import requests
import numpy as np
import soundfile as sf
from pyrubberband import pyrb
from dotenv import load_dotenv
load_dotenv()

IAM_TOKEN = os.getenv("YANDEX_IAM_TOKEN")
FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

TTS_API_URL = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"

EMOTION_SETTINGS = {
    "Нейтрально": {"voice": "zahar", "emotion": "neutral"},
    "Раздражение":   {"voice": "jane", "emotion": "evil"},
    "Радость":       {"voice": "jane", "emotion": "good"},
    "Дружелюбие":    {"voice": "marina",   "emotion": "friendly"},
}


def synthesize_speech(text, emotion_label="нейтральность", sample_rate=48000, output_file="output.wav"):
    if emotion_label not in EMOTION_SETTINGS:
        raise ValueError(f"Недопустимая эмоция: {emotion_label}. Доступны: {list(EMOTION_SETTINGS)}")

    voice_settings = EMOTION_SETTINGS[emotion_label]

    headers = {
        "Authorization": f"Bearer {IAM_TOKEN}",
    }

    data = {
        "text": text,
        "lang": "ru-RU",
        "folderId": FOLDER_ID,
        "voice": voice_settings["voice"],
        "emotion": voice_settings["emotion"],
        "speed": "1.0",
        "format": "lpcm",
        "sampleRateHertz": sample_rate
    }

    response = requests.post(TTS_API_URL, headers=headers, data=data)

    if response.status_code == 200:
        pcm_data = response.content
        audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        sf.write(output_file, audio_np, sample_rate)
        return output_file
    else:
        raise RuntimeError(f"Ошибка TTS API {response.status_code}: {response.text}")


def change_speed(input_path, rate=1.5, output_path="output_stretched.wav"):
    y, sr = sf.read(input_path)
    y_stretched = pyrb.time_stretch(y, sr, rate)
    sf.write(output_path, y_stretched, sr)
    return output_path


def apply_voice_conversion(input_path, output_path="converted.wav", speaker_id="ref_speaker"):
    logs_dir = "C:/Users/morik/PycharmProjects/VoiceSynthesisLLM/so-vits-svc-fork/logs/44k"

    # Поиск последнего файла G_*.pth
    g_paths = glob.glob(os.path.join(logs_dir, "G_*.pth"))
    if not g_paths:
        raise FileNotFoundError("Не найден ни один файл G_*.pth в logs/44k")

    # Отсортировать по номеру и взять последний
    g_latest = max(g_paths, key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))

    config_path = os.path.join(logs_dir, "config.json")

    command = [
        "svc",
        "infer",
        input_path,
        "--speaker", speaker_id,
        "--model-path", g_latest,
        "--config-path", config_path,
        "--output-path", output_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return output_path
    else:
        raise RuntimeError("Ошибка в voice conversion")

def convert_wav_to_mp3(input_wav, output_mp3="converted.mp3", remove_wav=False):
    ffmpeg_command = [
        "ffmpeg", "-y",
        "-i", input_wav,
        "-codec:a", "libmp3lame",
        "-qscale:a", "2",
        output_mp3
    ]
    subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if remove_wav:
        os.remove(input_wav)
    return output_mp3


def main():
    text = input("Введите текст для синтеза: ")
    emotion = input("Введите эмоцию: радость/нейтральность/дружелюбие/раздражение\n")
    speed = float(input("Задайте скорость: "))

    tts_file = synthesize_speech(text, emotion_label=emotion)
    stretched_file = apply_voice_conversion(tts_file)
    speed_file = change_speed(stretched_file, rate=speed)
    final_file = convert_wav_to_mp3(speed_file, os.path.splitext(speed_file)[0] + ".mp3")


if __name__ == "__main__":
    main()
