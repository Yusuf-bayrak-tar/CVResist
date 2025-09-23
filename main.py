from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2 as cv
import numpy as np
from resistance import detect_bands, preprocess_image

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-resistor/")
async def analyze_resistor(file: UploadFile = File(...)):
    try:
        # Görüntüyü oku
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv.imdecode(np_arr, cv.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Geçersiz görsel format",
                    "colors": []
                }
            )

        # Bandları tespit et
        bands = detect_bands(img)
        
        # Sadece renk bilgilerini hazırla
        colors = []
        for _, name, _, _, _ in bands:
            colors.append(name)

        return {
            "success": True,
            "message": "Renkler başarıyla tespit edildi",
            "colors": colors
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Hata oluştu: {str(e)}",
                "colors": []
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
