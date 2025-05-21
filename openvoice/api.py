from fastapi import FastAPI, HTTPException
import logging
from pydantic import BaseModel
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
import langid
import os

app = FastAPI()

class PredictRequest(BaseModel):
    prompt: str
    style: str
    audio_file_path: str
    agree: bool

# Load models and configurations
en_ckpt_base = 'checkpoints/base_speakers/EN'
zh_ckpt_base = 'checkpoints/base_speakers/ZH'
ckpt_converter = 'checkpoints/converter'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs'

# Load TTS and converter models
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
zh_base_speaker_tts = BaseSpeakerTTS(f'{zh_ckpt_base}/config.json', device=device)
zh_base_speaker_tts.load_ckpt(f'{zh_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Load speaker embeddings
en_source_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
zh_source_se = torch.load(f'{zh_ckpt_base}/zh_default_se.pth').to(device)

# Reference audio for fixed tone extraction
en_ref_audio = 'resources/en_reference.wav'
zh_ref_audio = 'resources/zh_reference.wav'

supported_languages = ['zh', 'en']

@app.post("/predict")
async def predict(request: PredictRequest):
    language_predicted = langid.classify(request.prompt)[0].strip()

    if language_predicted not in supported_languages:
        raise HTTPException(status_code=400, detail=f"Language '{language_predicted}' is not supported.")

    # Choose models and settings
    if language_predicted == "zh":
        tts_model = zh_base_speaker_tts
        source_se = zh_source_se
        ref_audio = zh_ref_audio
    else:
        tts_model = en_base_speaker_tts
        source_se = en_source_se
        ref_audio = en_ref_audio

    if not os.path.exists(ref_audio):
        raise HTTPException(status_code=500, detail="Reference audio file not found.")

    try:
        target_se, audio_name = se_extractor.get_se(ref_audio, tone_color_converter, target_dir='processed', vad=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting tone color: {str(e)}")

    # Generate source audio
    src_path = f'{output_dir}/tmp.wav'
    tts_model.tts(request.prompt, src_path, speaker='default', language=language_predicted)

    # Convert to target tone
    save_path = f'{output_dir}/output.wav'
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message="@MyShell"
    )

    return {
        "message": "Voice generated successfully.",
        "output_audio_path": save_path,
        "language": language_predicted
    }
