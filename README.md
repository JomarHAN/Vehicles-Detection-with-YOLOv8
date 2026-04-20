# Vehicle Detection

A small end-to-end showcase for **object detection** in a driving context: upload a street or dash-cam image and get **bounding boxes**, **class labels**, and **confidence scores** from a custom **YOLOv8** model, with a dark-themed web UI and a FastAPI backend.

This repo is meant for **cloning, running locally, and portfolio / demo use** — it pairs a trained detector with a simple API and frontend so others can try it without digging through notebooks first.

---

## Features

- **Web app** — Drag-and-drop or click to upload images; adjustable **confidence** and **IoU** thresholds; results show an annotated image plus a per-detection table (class, score, bbox).
- **REST API** — `POST /detect` accepts image uploads and returns JSON (base64 annotated image, detections, summary stats).
- **Notebook** — `notebooks/Autonomous_Driving_Part1.ipynb` documents the data-prep and training workflow (e.g. Colab-oriented paths); use it as a reference if you want to retrain or adapt the pipeline.

---

## Tech stack

| Layer | Technologies |
|--------|----------------|
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/) |
| **ML** | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [OpenCV](https://opencv.org/) (headless), [NumPy](https://numpy.org/), [Pillow](https://python-pillow.org/) |
| **Frontend** | Static HTML, CSS, vanilla JavaScript (no build step) |
| **API** | OpenAPI / automatic docs at `/docs` when the server is running |

Python dependencies are pinned in [`requirements.txt`](requirements.txt).

---

## Repository layout

```
autonomous_driving_demo/
├── main.py              # FastAPI app: /detect + static frontend
├── best.pt              # YOLO weights (required at runtime — see below)
├── requirements.txt
├── frontend/
│   └── index.html       # UI served at /
└── notebooks/
    └── Autonomous_Driving_Part1.ipynb
```

---

## Prerequisites

- **Python 3.10+** (3.11 recommended; match what you use for local dev).
- **`best.pt`** — A compatible Ultralytics YOLO checkpoint placed in the **project root** (same folder as `main.py`). The app loads it at startup. If you train your own model, export or copy weights to `best.pt`, or adjust `MODEL_PATH` in `main.py`.

---

## Quick start

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd autonomous_driving_demo
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Add `best.pt`** to the project root if it is not already there.

4. **Run the server** from the project root (so `frontend/` and `best.pt` resolve correctly):

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Open the app** — In a browser, go to [http://127.0.0.1:8000](http://127.0.0.1:8000).  
   Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## API overview

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/detect` | Multipart form: `file` (image). Query params: `conf`, `iou` (optional). Returns annotated image (base64 JPEG), detection list, and summary. |

The bundled frontend calls `/detect` on the **same origin** as the page (`window.location.origin`), so no extra CORS setup is needed when you use the built-in static server.

---

## Notes for GitHub / large files

- **`best.pt`** is often large. If you do not commit weights, document in your repo (or Releases) where to obtain them, or use [Git LFS](https://git-lfs.github.com/) if you choose to version the file.
- Virtual environments (`venv/`, `.venv/`) and training artifacts are listed in [`.gitignore`](.gitignore).

---

## License

Add a `LICENSE` file if you want to specify terms for your showcase; until then, assume **all rights reserved** unless you state otherwise.
