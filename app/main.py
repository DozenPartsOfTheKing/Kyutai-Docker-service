from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Optional, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.openapi.utils import get_openapi

from huggingface_hub import list_repo_files

# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

tags_metadata = [
    {
        "name": "health",
        "description": "Простой liveness-probe для Kubernetes/NGINX. Возвращает `ok`.",
    },
    {
        "name": "stt",
        "description": "**Speech-to-Text** на базе Kyutai STT. Принимает аудио-файл, возвращает транскрипт с тайм-стемпами.",
    },
    {
        "name": "tts",
        "description": "**Text-to-Speech**. Генерирует WAV/PCM с указанным голосом и температурой.",
    },
    {
        "name": "voices",
        "description": "Справочный энд-пойнт. Показывает доступные образцы голоса (Expresso dataset).",
    },
]

logger = logging.getLogger("server")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(
    title="SpeakerMan API",
    version="0.1.0",
    description="API-обёртка над Kyutai STT/TTS с поддержкой GPU и выбором голоса.",
    openapi_tags=tags_metadata,
)

# Allow all origins by default; tune if exposing publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"

# Constants for voice emotions we know exist in the Expresso dataset on HF.
Emotion = Literal["neutral", "happy", "angry", "sad"]


def _pick_voice_by_emotion(
    emotion: str,
    repo: str = "kyutai/tts-voices",
) -> str:
    """Return a relative voice path that matches the requested emotion.

    We scan files once (cached) and pick a deterministic voice sample for the
    given *emotion*. Raises *ValueError* if nothing is found.
    """

    # Cache results so we do not hit the network on every request.
    global _EMOTION_CACHE
    try:
        cache = _EMOTION_CACHE  # type: ignore[name-defined]
    except NameError:
        cache = {}  # type: ignore[assignment]
        _EMOTION_CACHE = cache  # type: ignore[misc]

    if repo not in cache:
        files = list_repo_files(repo)
        cache[repo] = [f for f in files if f.startswith("expresso/") and f.endswith(".wav")]

    matches = [f for f in cache[repo] if f"_{emotion}_" in f]
    if not matches:
        raise ValueError(f"No voice sample with emotion '{emotion}' found in {repo}.")
    # Pick the first file for stability.
    return sorted(matches)[0]


# Reuse the same cache for listing
def _list_voices(repo: str = "kyutai/tts-voices") -> list[str]:
    """Return all .wav voice sample paths inside *repo* under expresso/.*"""

    global _EMOTION_CACHE  # type: ignore[name-defined]
    try:
        cache = _EMOTION_CACHE  # type: ignore[name-defined]
    except NameError:
        cache = {}  # type: ignore[assignment]
        _EMOTION_CACHE = cache  # type: ignore[misc]

    if repo not in cache:
        files = list_repo_files(repo)
        cache[repo] = [f for f in files if f.startswith("expresso/") and f.endswith(".wav")]
    return cache[repo]


@app.get("/health", response_class=PlainTextResponse)
async def health() -> str:
    """Simple liveness probe."""
    return "ok"


async def _run_script(args: list[str]) -> str:
    """Runs a script asynchronously and captures stdout on success.

    Raises HTTPException on non-zero exit codes with stderr message.
    """
    logger.debug("Executing command: %s", " ".join(args))
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(ROOT_DIR),
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.error("Command failed: %s", stderr.decode())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=stderr.decode() or "Unknown error",
        )
    return stdout.decode()


@app.post("/stt", summary="Speech-to-Text", response_class=PlainTextResponse)
async def stt_endpoint(
    audio: Annotated[UploadFile, File(description="Аудио-файл для транскрипции")],
    hf_repo: Annotated[Optional[str], Form()] = None,
    vad: Annotated[bool, Form()] = False,
    device: Annotated[str, Form()] = "cuda",
) -> str:
    """Returns transcription with word-level timestamps using Kyutai STT.

    Internally calls `scripts/stt_from_file_pytorch.py`.
    """
    suffix = Path(audio.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = Path(tmp.name)

    args = [
        "python",
        str(SCRIPTS_DIR / "stt_from_file_pytorch.py"),
        str(tmp_path),
        "--device",
        device,
    ]
    if hf_repo:
        args += ["--hf-repo", hf_repo]
    if vad:
        args.append("--vad")

    try:
        transcript = await _run_script(args)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Failed to clean temp file: %s", exc)

    return transcript


@app.post("/tts", summary="Text-to-Speech")
async def tts_endpoint(
    text: Annotated[str, Form(description="Текст для синтеза")],
    hf_repo: Annotated[Optional[str], Form()] = "kyutai/tts-1.6b-en_fr",
    voice_repo: Annotated[Optional[str], Form()] = None,
    voice: Annotated[Optional[str], Form()] = None,
    emotion: Annotated[Optional[Emotion], Form(description="Эмоция голоса")] = None,
    device: Annotated[str, Form()] = "cuda",
    format: Annotated[str, Form(description="Формат аудио: wav или pcm")] = "wav",
    temp: Annotated[float, Form(description="Температура выборки (интонация), 0.0-1.0")] = 0.6,
) -> FileResponse:
    """Generates speech from text using Kyutai TTS.

    Returns the audio as WAV file.
    """
    # Map placeholder values from Swagger UI ("string") to None for easier use.
    if hf_repo in {None, "", "string"}:
        hf_repo = "kyutai/tts-1.6b-en_fr"
    if voice_repo in {None, "", "string"}:
        voice_repo = None
    if voice in {None, "", "string"}:
        voice = None
    if emotion in {None, "", "string"}:
        emotion = None

    # If emotion is provided but explicit voice is not, try to resolve it automatically.
    if emotion and not voice:
        try:
            resolved_repo = voice_repo or "kyutai/tts-voices"
            voice = _pick_voice_by_emotion(emotion, resolved_repo)
            voice_repo = resolved_repo
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # Write text to a temporary file because existing script expects a file or stdin
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as txt_file:
        txt_file.write(text)
        txt_path = Path(txt_file.name)

    suffix = ".wav" if format == "wav" else ".pcm"
    out_path = Path(tempfile.gettempdir()) / f"tts_{uuid.uuid4().hex}{suffix}"

    args: list[str] = [
        "python",
        str(SCRIPTS_DIR / "tts_pytorch.py"),
        str(txt_path),
        str(out_path),
        "--device",
        device,
    ]
    if hf_repo:
        args += ["--hf-repo", hf_repo]
    if voice_repo:
        args += ["--voice-repo", voice_repo]
    if voice:
        args += ["--voice", voice]
    # audio options
    args += ["--format", format, "--temp", str(temp)]

    try:
        await _run_script(args)
    finally:
        try:
            txt_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Failed to clean temp text: %s", exc)

    media_type = "audio/wav" if format == "wav" else "application/octet-stream"
    return FileResponse(
        path=out_path,
        filename=f"output.{format}",
        media_type=media_type,
    )


@app.get("/voices", summary="List available voice samples")
async def voices_endpoint(
    voice_repo: Annotated[str, Query(description="HF repo or local dir with voices")] = "kyutai/tts-voices",
    emotion: Annotated[Optional[str], Query(description="Filter by emotion substring e.g. happy, angry")] = None,
) -> list[str]:
    """Return list of available voice sample paths.

    This helps Swagger users pick a specific voice for the /tts endpoint.
    """

    try:
        voices = _list_voices(voice_repo)
        if emotion:
            voices = [v for v in voices if f"_{emotion}_" in v]
        return voices
    except Exception as exc:
        logger.error("Failed to list voices: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) 