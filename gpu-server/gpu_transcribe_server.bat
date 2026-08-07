@echo off
title GPU Server — Transcription + Summary Pipeline
cd /d "%~dp0"
set OLLAMA_MODELS=D:\OllamaModels

echo Starting Ollama...
start "" "C:\Users\spodg\AppData\Local\Programs\Ollama\ollama.exe" serve

echo Waiting for Ollama to be ready...
:wait_ollama
timeout /t 2 /nobreak >nul
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 goto wait_ollama
echo Ollama ready.

python gpu_server.py
pause
