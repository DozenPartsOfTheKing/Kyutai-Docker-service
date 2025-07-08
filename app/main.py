from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

logger = logging.getLogger("server")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(title="SpeakerMan API", version="0.1.0")

# Allow all origins by default; tune if exposing publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"


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
    device: Annotated[str, Form()] = "cuda",
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

    # Write text to a temporary file because existing script expects a file or stdin
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as txt_file:
        txt_file.write(text)
        txt_path = Path(txt_file.name)

    out_path = Path(tempfile.gettempdir()) / f"tts_{uuid.uuid4().hex}.wav"

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

    try:
        await _run_script(args)
    finally:
        try:
            txt_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Failed to clean temp text: %s", exc)

    return FileResponse(
        path=out_path,
        filename="output.wav",
        media_type="audio/wav",
    ) 