import tkinter as tk
from tkinter import ttk, messagebox
from synthesize import synthesize_speech, change_speed, apply_voice_conversion, convert_wav_to_mp3
import threading
import subprocess
import os
import sys

# Путь к папке с голосами преподавателей
TEACHER_DIR = r"C:\Users\morik\PycharmProjects\VoiceSynthesisLLM\so-vits-svc-fork\dataset_raw"
train_process = None  # для остановки обучения

def get_teacher_list():
    try:
        folders = [f for f in os.listdir(TEACHER_DIR) if os.path.isdir(os.path.join(TEACHER_DIR, f))]
        display_names = ["Холод И.И." if f == "ref_speaker" else f for f in folders]
        return display_names
    except Exception as e:
        print(f"Не удалось получить список преподавателей: {e}")
        return ["Холод И.И."]


def synthesize():
    text = text_input.get("1.0", tk.END).strip()
    emotion = emotion_var.get()
    speed = speed_var.get()
    teacher = teacher_var.get()

    if not text:
        messagebox.showwarning("Ошибка", "Введите текст для синтеза.")
        return

    threading.Thread(target=run_pipeline, args=(text, emotion, speed, teacher)).start()


def run_pipeline(text, emotion, speed, teacher_display_name):
    try:
        progress = tk.Toplevel(root)
        progress.title("Обработка")
        progress.geometry("300x120")
        progress_label = tk.Label(progress, text="Начинаем...", font=("Arial", 12))
        progress_label.pack(pady=30)
        progress.grab_set()

        progress_label.config(text="Синтез речи...")
        root.update()
        tts_file = synthesize_speech(text, emotion_label=emotion)

        teacher_folder = "ref_voice" if teacher_display_name == "Холод И.И." else teacher_display_name
        print(f"Преподаватель выбран: {teacher_folder}")

        progress_label.config(text="Клонирование голоса...")
        root.update()
        converted = apply_voice_conversion(tts_file)

        progress_label.config(text="Изменение скорости...")
        root.update()
        stretched = change_speed(converted, rate=speed)

        progress_label.config(text="Сохранение в MP3...")
        root.update()
        mp3_file = convert_wav_to_mp3(stretched)

        progress_label.config(text="Готово.")
        root.update()
        progress.after(1500, progress.destroy)
        messagebox.showinfo("Успех", f"Файл сохранён: {mp3_file}")
    except Exception as e:
        messagebox.showerror("Ошибка", str(e))


def open_train_log_window(mp3_path, speaker_name):
    global train_process

    log_window = tk.Toplevel(root)
    log_window.title("Обучение модели")
    log_window.geometry("600x400")

    log_text = tk.Text(log_window, wrap="word")
    log_text.pack(fill="both", expand=True)

    scrollbar = tk.Scrollbar(log_text)
    scrollbar.pack(side="right", fill="y")
    log_text.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=log_text.yview)

    stop_button = tk.Button(log_window, text="Остановить обучение", bg="red", fg="white")
    stop_button.pack(pady=5)

    def stream_output():
        global train_process
        train_process = subprocess.Popen(
            [sys.executable, "train_new_voice.py", mp3_path, speaker_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1
        )
        for line in train_process.stdout:
            log_text.insert("end", line)
            log_text.see("end")

        train_process.wait()
        log_text.insert("end", "\nОбучение завершено.")
        log_text.see("end")

    def stop_training():
        if train_process and train_process.poll() is None:
            train_process.terminate()
            log_text.insert("end", "\nОбучение остановлено пользователем.")
            log_text.see("end")

    stop_button.config(command=stop_training)

    threading.Thread(target=stream_output).start()


def open_add_teacher_window():
    window = tk.Toplevel(root)
    window.iconbitmap("leti_logo.ico")
    window.title("Добавить преподавателя")
    window.geometry("400x250")

    tk.Label(window, text="Путь к аудио (MP3, минимум 10 минут):").pack(pady=5)
    audio_entry = tk.Entry(window, width=50)
    audio_entry.pack()

    tk.Label(window, text="ФИО преподавателя:").pack(pady=5)
    name_entry = tk.Entry(window, width=50)
    name_entry.pack()

    warning_label = tk.Label(window, text="", fg="red")
    warning_label.pack(pady=5)

    def train_new_voice():
        mp3_path = audio_entry.get().strip()
        name = name_entry.get().strip()

        if not os.path.isfile(mp3_path):
            warning_label.config(text="Файл не найден.")
            return
        if not name:
            warning_label.config(text="Введите ФИО.")
            return

        import librosa
        try:
            y, sr = librosa.load(mp3_path, sr=None)
            duration_min = len(y) / sr / 60
            if duration_min < 10:
                warning_label.config(text=f"Аудио слишком короткое ({duration_min:.2f} мин), минимум 10 мин.")
                return
        except Exception:
            warning_label.config(text="Ошибка чтения аудио.")
            return

        window.destroy()
        open_train_log_window(mp3_path, name)

    tk.Button(window, text="Обучить", command=train_new_voice, bg="#2196F3", fg="white", padx=10, pady=5).pack(pady=10)


# --- Интерфейс ---
root = tk.Tk()
root.title("Синтез речи")
root.geometry("600x500")
root.iconbitmap("leti_logo.ico")

# Эмоции
tk.Label(root, text="Выберите эмоцию:").pack(pady=5)
emotion_var = tk.StringVar(value="Нейтрально")
emotion_menu = ttk.Combobox(root, textvariable=emotion_var, state="readonly", width=30)
emotion_menu['values'] = ["Нейтрально", "Раздражение", "Радость", "Дружелюбие"]
emotion_menu.pack()

# Скорость
tk.Label(root, text="Скорость речи:").pack(pady=5)
speed_var = tk.DoubleVar(value=1.0)
speed_slider = tk.Scale(root, from_=0.1, to=2.0, orient="horizontal", resolution=0.1, variable=speed_var, length=400)
speed_slider.pack()

# Преподаватель
tk.Label(root, text="Выберите преподавателя:").pack(pady=5)
teacher_var = tk.StringVar()
teacher_list = get_teacher_list()
teacher_menu = ttk.Combobox(root, textvariable=teacher_var, values=teacher_list, state="readonly", width=30)
teacher_menu.set(teacher_list[0])  # По умолчанию первый
teacher_menu.pack()

# Кнопка возле выбора преподавателя
tk.Button(root, text="Нет нужного", command=open_add_teacher_window, bg="#FFC107", padx=5).pack()

# Текст
tk.Label(root, text="Введите текст:").pack(pady=5)
text_input = tk.Text(root, height=8, wrap="word")
text_input.pack(padx=10, fill="both", expand=True)

# Кнопка
tk.Button(root, text="Синтезировать речь", command=synthesize, bg="#4CAF50", fg="white", padx=10, pady=5).pack(pady=10)

root.mainloop()
