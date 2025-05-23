import tensorflow as tf
from tacotron2.text import symbols

from types import SimpleNamespace
from tacotron2.text import symbols

from types import SimpleNamespace
from tacotron2.text import symbols


def create_hparams():
    """Create model hyperparameters."""
    hparams = SimpleNamespace(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files='filelists/ljs_audio_text_train_filelist.txt',
        validation_files='filelists/ljs_audio_text_val_filelist.txt',
        text_cleaners=['basic_cleaners'],  # Замените на русские cleaners

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=80.0,
        mel_fmax=11025.0,
        mel_mean=-5.0,
        mel_std=2.0,

        ################################
        # Model Parameters             #
        ################################

        symbols=list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя .,!?-:;'\""),  # + punctuation
        n_symbols=len(symbols),
        symbols_embedding_dim=256,

        # Encoder parameters
        encoder_input_dim=512,
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,
        decoder_rnn_dim=768,
        prenet_dim=256,
        max_decoder_steps=300,
        gate_threshold=0.2,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.2,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=256,
        attention_constraint=False,
        attention_windowing = False,

        # Location Layer parameters
        attention_location_n_filters=16,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=3,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-5,
        weight_decay=1e-6,
        grad_clip_thresh=0.5,
        batch_size=4,
        mask_padding=True,

        ################################
        # Emotion Parameters           #
        ################################
        n_emotions=7,  # Количество уникальных эмоций в вашем датасете
        emotion_embedding_dim=128,  # Размерность embedding для эмоций
        emotion_scale=1.0,
        emotion_learning_rate=1e-4,

        speaker_embedding_dim=128,  # Размерность эмбеддинга голоса
        contrastive_loss_weight=0.1,  # Вес контрастивного лосса

        teacher_forcing_start=1.0,
        teacher_forcing_final = 0.4,
        teacher_forcing_decay_epochs = 40,

        go_frame_value=-5.0,

        coverage_weight=1.0,

    )

    return hparams