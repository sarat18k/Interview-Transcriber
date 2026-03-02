🎤 Interview Audio Transcription & Analysis

An AI-powered Streamlit application that transcribes interview audio (with speaker diarization) and generates detailed candidate performance analysis using GPT-4.

This tool helps recruiters, hiring managers, and technical interviewers automatically evaluate interviews with structured AI-generated reports.

🚀 Features
🎧 Audio Input Options

Upload audio files (MP3, WAV, M4A)

Paste YouTube or audio/video links (auto-download via yt-dlp)

🗣 Speaker Diarization

Uses AssemblyAI for:

Accurate transcription

Speaker identification (Speaker 1, Speaker 2, etc.)

Clean, punctuated formatting

🧠 AI-Powered Analysis (GPT-4)

Choose from multiple analysis types:

Skill Summary

Communication skills

Domain knowledge

Confidence & clarity

Soft skills

Scorecard & hiring recommendation

Behavioral Analysis

Leadership

Adaptability

Collaboration

Behavioral patterns & red flags

Technical Depth

Technical clarity

Problem-solving approach

Practical knowledge

Tech rating

Extract Q&A

Extracts technical questions & answers

Rates each Q&A pair

Provides feedback and total rating

Custom Prompt

Ask any custom analysis question on the transcript

💾 Report Management

Transcript caching (avoids reprocessing same file)

Saves reports to interview_reports.csv

Downloadable:

Full AI analysis report

Full transcript

🏗 Tech Stack

Frontend/UI: Streamlit

Transcription & Diarization: AssemblyAI

AI Analysis: OpenAI GPT-4

Audio Download: yt-dlp

Data Handling: Pandas
