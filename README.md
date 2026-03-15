# Face Recognition App

Real-time face recognition application built with Python. Captures webcam video, detects faces with **MediaPipe** (468 landmarks), generates 128D encodings with **face_recognition** (dlib), and manages a database of known faces via a **PySide6** GUI.

## Stack

- **OpenCV** – camera capture and image processing  
- **MediaPipe** – face landmarks (tesselation + contours)  
- **face_recognition** – face locations, encoding generation and matching  
- **PySide6** – Qt GUI  
- **NumPy** – encodings and data  
- **JSON** – face metadata index (`data/index.json`)

## File structure

```
face_recognition_app/
├── run.py              # Entry point (run this)
├── requirements.txt
├── data/               # Created automatically
│   ├── index.json      # Face metadata index
│   ├── face_landmarker.task   # MediaPipe model (downloaded on first run)
│   └── faces/         # One folder per person
│       └── [person_name]/
│           ├── face.jpg
│           └── embedding.npy
└── src/
    ├── camera_handler.py
    ├── face_detector.py
    ├── face_database.py
    └── main.py
```

## Setup (virtual environment)

On Windows, **Python 3.13** is recommended for a prebuilt dlib wheel.

1. Create and activate venv:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies (Windows Python 3.13: install dlib wheel first):

   ```powershell
   pip install "https://github.com/eddiehe99/dlib-whl/releases/download/v20.0.0/dlib-20.0.0-cp313-cp313-win_amd64.whl"
   pip install -r requirements.txt
   ```

3. First run: the app downloads the MediaPipe face landmarker model to `data/face_landmarker.task` automatically.

## Run

- **Double-click:** `run.bat` (uses venv automatically)  
- **PowerShell:** `.\run.ps1`  
- **From project root:** `python run.py`  
- **Cursor / VS Code:** Set interpreter to `.\venv\Scripts\python.exe`, then **F5** (launches `run.py`)

## Usage

1. **Live Feed** – start webcam; green boxes and facial landmarks (mesh) overlay detected faces; status bar shows match name and distance if the face is in the database.  
2. **Save Face** – with a face visible (live or uploaded image), click **Save Face**, enter the person’s name; face image and encoding are stored under `data/faces/`.  
3. **Upload Image** – load a photo; faces are detected and can be saved or matched.  
4. **Search Faces** – type in the search box to filter the saved-faces list.  
5. **Rename** – use the pencil button on a face in the list to change its display name.

Data is stored under `data/`: `index.json` (metadata) and `faces/[name]/face.jpg` plus `embedding.npy` per person.
