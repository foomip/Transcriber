from types import SimpleNamespace

from lib import transcription


class NullProgressTimer:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        return None


def test_fmt_ts_formats_hours_minutes_and_fractional_seconds():
    assert transcription.fmt_ts(4.5) == "00:00:04.50"
    assert transcription.fmt_ts(723.1) == "00:12:03.10"
    assert transcription.fmt_ts(3723.45) == "01:02:03.45"


def test_language_helpers_normalize_lookup_and_format_languages():
    assert transcription.normalize_language_code(" EN ") == "en"
    assert transcription.language_description(" PT ") == "Portuguese"
    assert transcription.language_description("zz") is None

    supported = transcription.format_supported_languages().splitlines()
    assert "  af - Afrikaans" in supported
    assert "  en - English" in supported
    assert supported == sorted(supported, key=lambda line: line.split(" - ", 1)[1].casefold())


def test_detect_device_uses_cuda_when_ctranslate2_sees_a_cuda_gpu(monkeypatch):
    monkeypatch.setattr(transcription.ctranslate2, "get_cuda_device_count", lambda: 1)
    monkeypatch.setattr(transcription.torch.cuda, "get_device_name", lambda _index: "Test GPU")

    assert transcription.detect_device() == ("cuda", "float16")


def test_detect_device_uses_cpu_for_rocm_faster_whisper_fallback(monkeypatch):
    monkeypatch.setattr(transcription.ctranslate2, "get_cuda_device_count", lambda: 0)
    monkeypatch.setattr(transcription.torch.version, "hip", "6.0", raising=False)
    monkeypatch.setattr(transcription.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(transcription.torch.cuda, "get_device_name", lambda _index: "ROCm GPU")

    assert transcription.detect_device() == ("cpu", "int8")


def test_detect_device_uses_cpu_when_no_gpu_is_available(monkeypatch):
    monkeypatch.setattr(transcription.ctranslate2, "get_cuda_device_count", lambda: 0)
    monkeypatch.setattr(transcription.torch.version, "hip", None, raising=False)
    monkeypatch.setattr(transcription.torch.cuda, "is_available", lambda: False)

    assert transcription.detect_device() == ("cpu", "int8")


def test_transcribe_audio_formats_segments_and_language_metadata(monkeypatch):
    created_models = []

    class FakeWhisperModel:
        def __init__(self, model_size, *, device, compute_type):
            self.model_size = model_size
            self.device = device
            self.compute_type = compute_type
            self.transcribe_calls = []
            created_models.append(self)

        def transcribe(self, audio, *, beam_size=5, language=None):
            self.transcribe_calls.append((audio, beam_size, language))
            segments = [
                SimpleNamespace(start=0.0, end=4.5, text=" Hello there "),
                SimpleNamespace(start=4.5, end=65.25, text="Next topic"),
            ]
            info = SimpleNamespace(language="en", language_probability=0.94)
            return segments, info

    monkeypatch.setattr(transcription, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(transcription, "ProgressTimer", NullProgressTimer)

    result = transcription.transcribe_audio(
        "meeting_20260528_1030.wav",
        "cpu",
        "int8",
        language="en",
    )

    assert created_models[0].model_size == transcription.WHISPER_MODEL_SIZE
    assert created_models[0].device == "cpu"
    assert created_models[0].compute_type == "int8"
    assert created_models[0].transcribe_calls == [("meeting_20260528_1030.wav", 5, "en")]
    assert result.lines == [
        "[00:00:00.00 -> 00:00:04.50]  Hello there",
        "[00:00:04.50 -> 00:01:05.25]  Next topic",
    ]
    assert result.requested_language == "en"
    assert result.requested_language_description == "English"
    assert result.detected_language == "en"
    assert result.detected_language_description == "English"
    assert result.language_probability == 0.94
