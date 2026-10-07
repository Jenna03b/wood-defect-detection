from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI(title="Wood Defect Detecion API")

model=YOLO("weights/best.pt")

@app.get("/")
def home():
    return {"message": "Wood Defect Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf_thresh: float = 0.25):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(source=img, conf=conf_thresh, imgsz=768)
    result = results[0]

    detections = []
    for box in result.boxes:
        detections.append({
            "class_id": int(box.cls[0]),
            "class_name": model.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "box": [float(x) for x in box.xyxy[0]]
        })

    return {
        "filename": file.filename,
        "count": len(detections),
        "detections": detections
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)