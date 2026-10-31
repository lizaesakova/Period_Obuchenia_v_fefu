#!/usr/bin/env python3
"""
Транскрибация голосовых сообщений из Telegram (.ogg / .oga / .opus, Opus).

Использование:
    uv sync && uv run main.py

Модели: tiny, base, small, medium, large-v3
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment
from faster_whisper import WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("transcriber")

# Ограничения на входные файлы
ALLOWED_EXTENSIONS = {".ogg", ".oga", ".opus"}
MAX_FILE_SIZE_MB = 100      # максимальный размер файла
MAX_DURATION_SEC = 600      # максимальная длительность (10 минут)

# Кэш моделей: каждая модель загружается один раз и переиспользуется
_MODELS: dict[str, WhisperModel] = {}


def get_model(model_size: str = "small") -> WhisperModel:
    """Загружает модель Whisper один раз и переиспользует при повторных вызовах."""
    if model_size not in _MODELS:
        logger.info("Загружаю модель Whisper: %s", model_size)
        _MODELS[model_size] = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )
    return _MODELS[model_size]


def clean_audio(audio_path: str, output_path: str, denoise: bool = False) -> str:
    """
    Подготовка аудио: OGG/Opus -> WAV 16 kHz mono 16-bit + нормализация.

    Агрессивное шумоподавление ВЫКЛЮЧено по умолчанию (denoise=False),
    т.к. оно может искажать речь. Включается только по запросу.
    """
    audio = AudioSegment.from_file(audio_path)
    audio = (
        audio
        .set_frame_rate(16000)
        .set_channels(1)
        .set_sample_width(2)
    )
    audio.export(output_path, format="wav")

    data, sample_rate = sf.read(output_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)

    if denoise:
        logger.info("Применяю шумоподавление...")
        data = nr.reduce_noise(
            y=data,
            sr=sample_rate,
            stationary=False,
            prop_decrease=0.6,
        )

    # Пиковая нормализация громкости
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * 0.9

    sf.write(output_path, data, sample_rate, subtype="PCM_16")
    return output_path


def _validate_input(file_path: str) -> None:
    """Проверяет существование файла, расширение и размер."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    ext = Path(file_path).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Неподдерживаемое расширение '{ext}'. "
            f"Ожидается: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"Файл слишком большой: {size_mb:.1f} МБ "
            f"(лимит {MAX_FILE_SIZE_MB} МБ)"
        )


def transcribe(
    file_path: str,
    model_size: str = "small",
    language: str | None = None,
    denoise: bool = False,
) -> str:
    """
    Основная функция: очистка звука + транскрибация.

    Args:
        file_path: путь к файлу .ogg / .oga / .opus
        model_size: модель Whisper (tiny/base/small/medium/large-v3)
        language: язык распознавания; None — Whisper определит сам
        denoise: включить шумоподавление (по умолчанию выключено)

    Returns:
        Транскрибированный текст.
    """
    _validate_input(file_path)

    model = get_model(model_size)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        cleaned_path = tmp.name

    try:
        logger.info("Очищаю звук: %s", file_path)
        clean_audio(file_path, cleaned_path, denoise=denoise)

        # Проверка длительности по заголовку файла, без загрузки в память
        wav_info = sf.info(cleaned_path)
        if wav_info.duration > MAX_DURATION_SEC:
            raise ValueError(
                f"Аудио слишком длинное: {wav_info.duration:.0f} сек "
                f"(лимит {MAX_DURATION_SEC} сек)"
            )

        logger.info("Транскрибирую...")

        audio_array, sr = sf.read(cleaned_path, dtype="float32")
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        segments, info = model.transcribe(
            audio_array,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )

        logger.info(
            "Язык: %s (уверенность %.1f%%), длительность %.1f сек",
            info.language,
            info.language_probability * 100,
            info.duration,
        )

        return " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        )

    finally:
        if os.path.exists(cleaned_path):
            os.unlink(cleaned_path)


def main() -> None:
    if len(sys.argv) < 2:
        print("Использование: uv run main.py <файл.ogg> [модель] [язык] [--denoise]")
        print("Пример:        uv run main.py audio\\song.ogg small ru")
        print("Язык: ru, en, ... либо auto (по умолчанию — автоопределение)")
        sys.exit(1)

    file_path = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "small"
    language = sys.argv[3] if len(sys.argv) > 3 else None
    if language in ("auto", "none", "-"):
        language = None
    denoise = "--denoise" in sys.argv

    try:
        text = transcribe(
            file_path,
            model_size=model_size,
            language=language,
            denoise=denoise,
        )
    except (FileNotFoundError, ValueError) as e:
        # Ожидаемые ошибки пользователю — коротко и понятно, без стектрейса
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        # Всё остальное — подробно в лог, пользователю — коротко
        logger.exception("Транскрибация завершилась с ошибкой")
        print(f"❌ Ошибка транскрибации: {e}")
        sys.exit(1)

    print("=" * 50)
    print("📄 РЕЗУЛЬТАТ ТРАНСКРИБАЦИИ:")
    print("=" * 50)
    print(text if text else "(речь не обнаружена)")
    print("=" * 50)


if __name__ == "__main__":
    main()