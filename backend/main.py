# backend/main.py

import os
from pathlib import Path
from typing import Union

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import groq

# --- KONFIGURASI APLIKASI ---

# Path Konfigurasi
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
TEMPLATES_DIR = FRONTEND_DIR / "templates"

# Konfigurasi API Groq
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
WHISPER_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "openai/gpt-oss-20b"

if not GROQ_API_KEY:
    raise ValueError("Environment variable GROQ_API_KEY tidak ditemukan. Silakan atur terlebih dahulu.")

# --- INISIALISASI KLIEN GROQ ---

try:
    groq_client = groq.Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Gagal menginisialisasi klien Groq: {e}")
    groq_client = None

# --- MODEL DATA & EXCEPTION ---

class AnalysisResponse(BaseModel):
    """Model respons untuk endpoint analisis audio."""
    analysis: Union[str, None] = None
    error: Union[str, None] = None

class ServiceError(Exception):
    """Kesalahan dasar untuk layanan internal."""
    pass

class GroqError(ServiceError):
    """Kesalahan saat berkomunikasi dengan API Groq."""
    pass

class TranscriptionError(ServiceError):
    """Kesalahan selama proses transkripsi audio."""
    pass

# --- LOGIKA BISNIS (LAYANAN) ---

def transcribe_audio(audio_file: UploadFile) -> str:
    """Mentranskripsi audio menggunakan model Whisper dari Groq.

    Args:
        audio_file: Objek UploadFile dari FastAPI.

    Returns:
        String hasil transkripsi.

    Raises:
        TranscriptionError: Jika transkripsi gagal atau hasil kosong.
        GroqError: Jika terjadi kesalahan API.
    """
    if not groq_client:
        raise GroqError("Klien Groq tidak tersedia.")

    try:
        transcription = groq_client.audio.transcriptions.create(
            file=(audio_file.filename, audio_file.file),
            model=WHISPER_MODEL,
            response_format="text",
            language="id" # Opsional: bisa diset ke 'id' untuk Bahasa Indonesia
        )

        if not transcription.strip():
            raise TranscriptionError("Tidak dapat mendeteksi suara dalam rekaman. Audio mungkin terlalu sunyi atau tidak jelas.")

        return transcription
    except groq.APIError as e:
        raise GroqError(f"Error dari API Groq saat transkripsi: {e}")
    except Exception as e:
        raise TranscriptionError(f"Gagal mentranskripsi audio: {e}")

def analyze_text_with_groq(text: str) -> str:
    """Menganalisis teks transkripsi menggunakan model LLM dari Groq.

    Args:
        text: Teks hasil transkripsi.

    Returns:
        String hasil analisis dari LLM.

    Raises:
        GroqError: Jika terjadi kesalahan API.
    """
    if not groq_client:
        raise GroqError("Klien Groq tidak tersedia.")

    system_prompt = (
    "[RULES]"
    "Always respond in Bahasa Indonesia."
    "DO NOT respond with markdown formatting."
    "IF there is no user_message or depression detected, respond with: 'Tidak ada indikasi depresi.'"
    "[END RULES]"
    "[INSTRUCTIONS]"
    "Your job is to analyze the given user messages for signs of depression."
    "You will provide a detailed analysis based on the following criteria:"
    "- Frequency of negative words and phrases."
    "- Presence of suicidal or self-harm thoughts."
    "- Emotional tone and intensity."
    "Your response should be clear, concise, and actionable."
    "[END INSTRUCTIONS]"
    )
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": f"Teks transkripsi:\n\"{text}\"",
                }
            ],
            model=LLM_MODEL,
        )
        return chat_completion.choices[0].message.content
    except groq.APIError as e:
        raise GroqError(f"Error dari API Groq saat analisis: {e}")
    except Exception as e:
        raise GroqError(f"Terjadi kesalahan saat menganalisis teks: {e}")

# --- INISIALISASI APLIKASI FASTAPI ---

app = FastAPI(
    title="Audio Depression Analyzer API (Groq)",
    description="API untuk menganalisis indikator depresi dari file audio menggunakan layanan Groq.",
    version="2.0.0"
)

# Konfigurasi Static Files dan Templates
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- ENDPOINT API ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Menyajikan halaman utama aplikasi (index.html)."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio_endpoint(audio: UploadFile = File(...)):
    """Endpoint untuk menganalisis audio yang diunggah menggunakan layanan Groq."""
    # Validasi tipe file
    if not audio.content_type or "audio" not in audio.content_type:
        raise HTTPException(status_code=400, detail="File yang diunggah bukan file audio yang valid.")

    try:
        transcription = transcribe_audio(audio)
        analysis = analyze_text_with_groq(transcription)
        return {"analysis": analysis}

    except TranscriptionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except GroqError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan tak terduga: {e}")

@app.on_event("startup")
async def startup_event():
    """Menampilkan pesan konfigurasi saat server berhasil dijalankan."""
    print("Aplikasi telah dimulai dengan layanan Groq.")
    print(f"Model Transkripsi: {WHISPER_MODEL}")
    print(f"Model LLM: {LLM_MODEL}")