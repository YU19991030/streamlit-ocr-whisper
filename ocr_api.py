from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from faster_whisper import WhisperModel
from PIL import Image
import numpy as np
import cv2
import io
import tempfile
import os

app = FastAPI()

# 啟用 CORS（允許前端跨域連線）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可依需求調整為前端網址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 PaddleOCR（Mobile 模型）與 Whisper（tiny）
ocr_model = PaddleOCR(use_angle_cls=True, lang='ch')  # 自動會抓 mobile 模型
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# ----------- 名片 OCR 路由 -----------
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

# ----------- Render 專用啟動入口 -----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("ocr_api:app", host="0.0.0.0", port=port)
