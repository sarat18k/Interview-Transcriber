import streamlit as st
import requests
import os
import pandas as pd
import time
import yt_dlp
import glob
from dotenv import load_dotenv
from datetime import datetime
import openai

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

openai.api_key = OPENAI_API_KEY

# Initialize session state
if 'transcripts' not in st.session_state:
    st.session_state['transcripts'] = {}

# AssemblyAI diarization function
def assemblyai_diarize(audio_bytes):
    headers_auth = {'authorization': ASSEMBLYAI_API_KEY}
    upload_response = requests.post('https://api.assemblyai.com/v2/upload', headers=headers_auth, data=audio_bytes)
    if upload_response.status_code != 200:
        st.error("Failed to upload audio to AssemblyAI")
        return None
    upload_url = upload_response.json()['upload_url']

    transcript_request = {
        "audio_url": upload_url,
        "speaker_labels": True,
        "auto_chapters": False,
        "iab_categories": False,
        "auto_highlights": False,
        "punctuate": True,
        "format_text": True
    }
    transcript_response = requests.post(
        'https://api.assemblyai.com/v2/transcript',
        headers={**headers_auth, 'content-type': 'application/json'},
        json=transcript_request
    )
    if transcript_response.status_code != 200:
        st.error("Failed to request transcription from AssemblyAI")
        return None
    transcript_id = transcript_response.json()['id']

    while True:
        status_response = requests.get(f'https://api.assemblyai.com/v2/transcript/{transcript_id}', headers=headers_auth)
        status = status_response.json()['status']
        if status == 'completed':
            break
        elif status == 'error':
            st.error("AssemblyAI transcription error: " + status_response.json().get('error', ''))
            return None
        time.sleep(3)

    paragraphs = status_response.json().get('utterances', [])
    transcript_text = ""
    for para in paragraphs:
        speaker = para.get('speaker', 'Speaker')
        text = para.get('text', '')
        transcript_text += f"{speaker}: {text}\n\n"
    return transcript_text.strip()


# Fallback Whisper transcription (commented out, not required)
# def transcribe_with_openai_whisper(audio_bytes):
#     with open("temp_audio.mp3", "wb") as f:
#         f.write(audio_bytes)
#     with open("temp_audio.mp3", "rb") as audio_file:
#         transcript = openai.audio.transcriptions.create(file=audio_file, model="whisper-1")
#     os.remove("temp_audio.mp3")
#     return transcript.text

# GPT-4 analysis function
def analyze_with_openai(prompt):
    return analyze_with_openai_max_tokens(prompt, max_tokens=1500)

def analyze_with_openai_max_tokens(prompt, max_tokens=1500):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a precise AI interview assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()

# Save report to CSV
def save_report_to_csv(filename, report_dict):
    try:
        df = pd.read_csv(filename)
        if (df['timestamp'] == report_dict['timestamp']).any():
            return
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([report_dict])], ignore_index=True)
    df.to_csv(filename, index=False)

# Streamlit UI setup
st.set_page_config(page_title="Interview Analyzer", layout="wide")
st.title("🎤 Interview Audio Transcription & Analysis")

if not OPENAI_API_KEY:
    st.error("OpenAI API key is missing. Please check your .env file.")
    st.stop()
if not ASSEMBLYAI_API_KEY:
    st.error("AssemblyAI API key is missing. Please check your .env file.")
    st.stop()

input_mode = st.radio("Choose input method", ("Upload File", "Paste Link"))

# Audio input handling
if input_mode == "Upload File":
    uploaded_file = st.file_uploader("Upload audio file (MP3/WAV/M4A)", type=["mp3", "wav", "m4a"])
    if uploaded_file:
        uploaded_file.seek(0)
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes)
        st.session_state['audio_bytes'] = audio_bytes
        st.session_state['identifier'] = uploaded_file.name

elif input_mode == "Paste Link":
    url = st.text_input("Paste video/audio link")
    if url and st.button("Fetch Audio"):
        try:
            st.info("Downloading audio from link...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_pattern = f"temp_{timestamp}.%(ext)s"
            ydl_opts = {'format': 'bestaudio/best', 'outtmpl': filename_pattern, 'quiet': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            actual_file = glob.glob(f"temp_{timestamp}.*")
            if not actual_file:
                raise FileNotFoundError("Audio file not found after yt-dlp processing.")
            audio_path = actual_file[0]
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes)
            st.session_state['audio_bytes'] = audio_bytes
            st.session_state['identifier'] = url
            os.remove(audio_path)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

analysis_type = st.selectbox("Select analysis type", [
    "Skill Summary", "Behavioral Analysis", "Technical Depth", "Extract Q&A", "Custom Prompt"
])

custom_prompt = ""
if analysis_type == "Custom Prompt":
    custom_prompt = st.text_area("Enter your custom prompt:")

# Prompt templates
prompt_map = {
    "Skill Summary": (
        "You are an expert interviewer evaluator.\n\n"
        "Analyze the following interview transcript in-depth and assess the candidate's performance based on the following categories:\n"
        "- Communication Skills\n"
        "- Domain Knowledge\n"
        "- Confidence and Clarity\n"
        "- Soft Skills (Teamwork, Leadership, Adaptability, etc.)\n\n"
        "Your output should include:\n"
        "1. A well-written summary of the candidate's performance (at least 3 detailed paragraphs)\n"
        "2. A bullet point list of strengths and areas of improvement\n"
        "3. A scorecard with ratings out of 10 for each category\n"
        "4. An overall recommendation (Hire, Consider, or Reject) with justification\n\n"
        "Format:\n"
        "**Candidate Performance Summary:**\n"
        "[Full detailed summary]\n\n"
        "**Strengths:**\n"
        "- [Strength 1]\n"
        "- [Strength 2]\n\n"
        "**Areas of Improvement:**\n"
        "- [Weakness 1]\n"
        "- [Weakness 2]\n\n"
        "**Scorecard:**\n"
        "- Communication: X/10\n"
        "- Domain Knowledge: X/10\n"
        "- Confidence & Clarity: X/10\n"
        "- Soft Skills: X/10\n"
        "- Overall Score: X/10\n\n"
        "**Recommendation:** [Hire / Consider / Reject] — [Justify in 1-2 sentences]"
    ),
    "Behavioral Analysis": (
        "You are a behavioral analysis expert specializing in professional interviews.\n\n"
        "Read the following transcript and provide a deep behavioral assessment of the candidate. Focus on:\n"
        "- Confidence and composure\n"
        "- Leadership and ownership\n"
        "- Adaptability and openness to feedback\n"
        "- Problem-solving and critical thinking\n"
        "- Team collaboration and communication\n\n"
        "Output Format:\n"
        "**Behavioral Analysis:**\n"
        "[3-4 detailed paragraphs highlighting key behavioral traits]\n\n"
        "**Behavioral Strengths:**\n"
        "- [Trait + example]\n"
        "- [Trait + example]\n\n"
        "**Concerns or Red Flags:**\n"
        "- [Trait + explanation]\n\n"
        "**Observed Behavioral Patterns:**\n"
        "[List any repeat behaviors, mindset patterns, or attitudes reflected in responses]"
    ),
    "Technical Depth": (
        "You are a senior technical interviewer.\n\n"
        "Evaluate the candidate's technical depth demonstrated in this interview transcript. Consider:\n"
        "- Clarity and depth of technical concepts explained\n"
        "- Problem-solving approach and accuracy\n"
        "- Practical knowledge and examples\n"
        "- Consistency in technical understanding\n\n"
        "Output Format:\n"
        "**Technical Assessment:**\n"
        "[3-4 detailed paragraphs with in-depth analysis of technical responses]\n\n"
        "**Technical Strengths:**\n"
        "- [Skill + example]\n"
        "- [Skill + example]\n\n"
        "**Technical Weaknesses or Gaps:**\n"
        "- [Area + reason]\n\n"
        "**Tech Rating (out of 10):** X/10\n"
        "**Recommendation:** [Strong / Moderate / Weak] Technical Candidate — [Optional comment]"
    ),
    "Extract Q&A": (
        "You are a professional transcript processor.\n\n"
        "From the following transcript, extract the all technical questions asked by the Interviewer and the corresponding answers from the Interviewee.\n"
        "Only include clearly technical questions (skip HR, personal, or small talk), If the questions arent techincal related please dont provide them.\n"
        "Do not generate or paraphrase. Use exact phrasing from the transcript when possible.\n\n"
        "Format:\n"
        "**Q1:** [Question text]\n\n"
        "**A1:** [Answer text — at least 2-4 sentences if possible]\n\n"
        "**Q2:** ...\n"
        "(up to end of the questions)\n\n"
        "Ensure that:\n"
        "- Each Q&A pair is useful and complete.\n"
        "- Answers are comprehensive and not clipped or overly short."
        "Also provide rating for each question & answer pair after each one and total rating at last along with feedback"
    )
}

if st.button("Submit"):
    if 'audio_bytes' not in st.session_state or 'identifier' not in st.session_state:
        st.error("No audio loaded. Please upload or fetch audio before submitting.")
        st.stop()

    audio_bytes = st.session_state['audio_bytes']
    identifier = st.session_state['identifier']

    if identifier in st.session_state['transcripts']:
        transcript_text = st.session_state['transcripts'][identifier]
        st.info("Loaded transcript from cache.")
    else:
        with st.spinner("Running speaker diarization with AssemblyAI..."):
            transcript_text = assemblyai_diarize(audio_bytes)


    # if not transcript_text:
    #     st.warning("AssemblyAI diarization failed, falling back to Whisper raw transcription.")
    #     transcript_text = transcribe_with_openai_whisper(audio_bytes)

        st.session_state['transcripts'][identifier] = transcript_text

    with st.expander("📄 View Transcript"):
        st.text_area("Transcript", transcript_text, height=400)

    if analysis_type == "Custom Prompt":
        formatted_prompt = f"""
You are a helpful AI. Based on the transcript below, answer the custom request.

### Transcript:
{transcript_text}

### Request:
{custom_prompt}

### Response:
"""
        max_tokens = 1500
    elif analysis_type == "Extract Q&A":
        base_prompt = prompt_map[analysis_type]
        formatted_prompt = f"""
You are a precise AI interview assistant.

### Transcript:
{transcript_text}

### Task:
{base_prompt}
"""
        max_tokens = 3000
    else:
        base_prompt = prompt_map[analysis_type]
        formatted_prompt = f"""
You are a precise AI interview assistant.

### Transcript:
{transcript_text}

### Task:
{base_prompt}
"""
        max_tokens = 1500

    st.info("Running analysis with OpenAI GPT-4...")
    analysis = analyze_with_openai_max_tokens(formatted_prompt, max_tokens=max_tokens)

    if analysis_type == "Custom Prompt":
        st.subheader("📝 Custom Request")
        st.markdown(f"> {custom_prompt}")
    st.subheader("🧠 AI Response")
    st.markdown(analysis)

    timestamp = datetime.now().isoformat()
    filename_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dict = {
        "timestamp": timestamp,
        "filename": identifier,
        "analysis_type": analysis_type,
        "transcript": transcript_text,
        "analysis": analysis,
        "prompt_used": custom_prompt if analysis_type == "Custom Prompt" else prompt_map[analysis_type]
    }
    save_report_to_csv("interview_reports.csv", report_dict)

    st.session_state['ready_for_download'] = {
        "analysis": analysis,
        "transcript": transcript_text,
        "filename_ts": filename_ts
    }

# Download buttons
if 'ready_for_download' in st.session_state:
    d = st.session_state['ready_for_download']
    st.download_button("📥 Download Report", d["analysis"], file_name=f"interview_analysis_{d['filename_ts']}.txt")
    st.download_button("📥 Download Transcript", d["transcript"], file_name=f"transcript_{d['filename_ts']}.txt")
