import os
from pydub import AudioSegment

# Путь к папке с mp3 файлами
input_folder = "C:/Users/morik/PycharmProjects/VoiceSynthesisLLM/data/voice"
# Путь к папке для сохранения wav файлов
output_folder = "C:/Users/morik/PycharmProjects/VoiceSynthesisLLM/data/wavs"

# Настройки конвертации
sample_rate = 16000  # Частота дискретизации (например, 16000 Гц)
channels = 1  # Количество каналов (1 для моно, 2 для стерео)

# Создаем папку для wav файлов, если её нет
os.makedirs(output_folder, exist_ok=True)

# Конвертация всех mp3 файлов в wav
for filename in os.listdir(input_folder):
    if filename.endswith(".mp3"):
        # Загружаем mp3 файл
        mp3_path = os.path.join(input_folder, filename)
        audio = AudioSegment.from_mp3(mp3_path)

        # Применяем настройки (частота дискретизации и каналы)
        audio = audio.set_frame_rate(sample_rate)  # Устанавливаем частоту дискретизации
        audio = audio.set_channels(channels)  # Устанавливаем количество каналов

        # Сохраняем в формате wav
        wav_filename = filename.replace(".mp3", ".wav")
        wav_path = os.path.join(output_folder, wav_filename)
        audio.export(wav_path, format="wav")

        print(f"Конвертирован: {filename} -> {wav_filename} (частота: {sample_rate} Гц, каналы: {channels})")

print("Конвертация завершена!")