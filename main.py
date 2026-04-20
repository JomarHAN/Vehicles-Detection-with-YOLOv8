import io, base64, cv2
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO

app = FastAPI(title="Vehicle Detection API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["POST"], allow_headers=["*"]
)

# Load model once at startup
MODEL_PATH = Path("best.pt")
if not MODEL_PATH.exists():
    raise RuntimeError("best.pt not found - make sure it's in the project root.")

print("Loading model...")
model = YOLO(str(MODEL_PATH))
CLASS_NAME = model.names
print(f"Model loaded. Classes: {CLASS_NAME}")


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.25, iou: float = 0.45):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could nor decode image")

    results = model.predict(
        source=img_bgr, imgsz=640, conf=conf, iou=iou, verbose=False
    )
    result = results[0]

    # Annotated image -> base64, never touches disk
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=88)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Build reponse
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls)
        detections.append(
            {
                "class": CLASS_NAME[cls_id],
                "confidence": round(float(box.conf), 4),
                "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
            }
        )

    class_counts = {}
    for d in detections:
        class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

    avg_conf = (
        round(sum(d["confidence"] for d in detections) / len(detections), 3)
        if detections
        else 0
    )

    return {
        "annotated_image_b64": img_b64,
        "detections": detections,
        "summary": {
            "total_detections": len(detections),
            "class_counts": class_counts,
            "avg_confidence": avg_conf,
        },
    }


# Serve frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
