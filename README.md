<div align="center"> 

  
  # EmotionalSpeedTTS

  
</div>


---

<div align="center">  
This is the final qualification work (Bachelor’s thesis) by Morik A.I.,  
student of group 1308 at Saint Petersburg Electrotechnical University "LETI".  
Scientific advisor — Timofeev A.V..  
2025.
</div>

---

EmotionalSpeedTTS is a Russian-language text-to-speech (TTS) system with support for emotion control, adjustable speaking rate, and speaker voice cloning. It is designed for applications such as adaptive educational systems, interactive learning content, and intelligent assistants.


This project is based on a modified version of Tacotron 2 and integrates:
- Emotion-conditioned speech synthesis
- Speed-controllable generation
= Voice cloning through So-VITS-SVC
- High-quality waveform generation using HiFi-GAN
- A graphical user interface for ease of use


### Features
- Emotional synthesis
- Adjustable speech speed (0.1× to 2×)
- Voice cloning for individual instructors via uploaded audio samples
- GUI interface for end-to-end text-to-speech workflow
- HiFi-GAN vocoder for realistic waveform synthesis
- Custom voice training workflow with automatic preprocessing


---


## Dependencies and Resources



So-VITS-SVC (voice conversion) https://github.com/voicepaw/so-vits-svc-fork


HiFi-GAN (neural vocoder) https://github.com/jik876/hifi-gan


RESD (Russian Emotional Speech Dataset) https://www.kaggle.com/datasets/ar4ikov/resd-dataset


Pretrained model storage https://disk.yandex.ru/d/Zlp0W7jzDPlFuQ


Tacotron 2 (voice synthesis) https://github.com/NVIDIA/tacotron2



---


## GUI Capabilities
1. Select emotion from dropdown
2. Set desired speaking speed via slider
3. Choose or add instructor voice
4. Upload your own audio sample (min. 10 minutes) for training
5. View synthesis and training progress

Custom Voice Training
Users can upload a .mp3 audio file (minimum 10 minutes) and enter the name of the speaker. The system will:
1. Convert the file to .wav (44.1 kHz, mono)
2. Prepare training data directory
3. Run svc preprocessing steps
Start training a new voice model automatically
