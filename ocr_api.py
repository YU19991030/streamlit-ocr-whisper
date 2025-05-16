from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from faster_whisper import WhisperModel
from PIL import Image
import numpy as np
import io
import cv2
import tempfile
import os

app = FastAPI()

# CORS 支援（讓 Streamlit 前端可跨網域連線）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化模型
ocr_model = PaddleOCR(use_angle_cls=True, lang='ch')  # Mobile 模型已內建
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# ✅ OCR API：圖片辨識
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print(f"📸 圖片大小：{image.size}")  # ✅ Debug：確認有圖
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result = ocr_model.ocr(img, cls=True)
        text = "\n".join([line[1][0] for box in result for line in box])
        return {"text": text}
    except Exception as e:
        print("❌ OCR 錯誤：", e)  # ✅ 顯示錯誤細節
        return {"error": str(e)}

# ✅ Whisper API：語音辨識
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

# ✅ 讓 Render 正常綁定 port
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("ocr_api:app", host="0.0.0.0", port=port)
