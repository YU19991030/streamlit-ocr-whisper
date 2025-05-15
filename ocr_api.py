from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from faster_whisper import WhisperModel
from PIL import Image
import numpy as np
import cv2
import io
tempfile

app = FastAPI()

# 允許跨域請求（給 Streamlit 用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 OCR & Whisper 模型
ocr_model = PaddleOCR(use_angle_cls=True, lang='ch')
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# ----------- OCR 路由（改為接收 UploadFile） -----------
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result = ocr_model.ocr(img, cls=True)
        text = "\n".join([line[1][0] for box in result for line in box])
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}

# ----------- Whisper 語音辨識路由 -----------
@app.post("/whisper")
async def whisper_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        segments, _ = whisper_model.transcribe(
            tmp_path, language="zh", beam_size=5, vad_filter=True
        )
        text = " ".join([seg.text.strip() for seg in segments])
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}
