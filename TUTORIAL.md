# Face Recognition App - Complete Build Tutorial

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites and Setup](#prerequisites-and-setup)
3. [File Structure](#file-structure)
4. [Understanding the Architecture](#understanding-the-architecture)
5. [Building the Project](#building-the-project)
   - [Step 1: Camera Handler](#step-1-camera-handler)
   - [Step 2: Face Detector](#step-2-face-detector)
   - [Step 3: Face Database](#step-3-face-database)
   - [Step 4: Main GUI Application](#step-4-main-gui-application)
   - [Step 5: Run Script](#step-5-run-script)
6. [Testing Your Application](#testing-your-application)

---

## Project Overview

This is a **real-time face recognition application** built with Python that can:
- Capture video from your webcam
- Detect faces in real-time using MediaPipe
- Generate unique face encodings using deep learning
- Save and recognize known faces
- Upload and process static images
- Manage a database of known faces with a GUI

**Core Technologies:**
- **OpenCV**: Camera capture and image processing
- **MediaPipe**: Fast face detection and facial landmarks (468 points per face)
- **face_recognition**: Face encoding generation and matching (based on dlib)
- **PySide6**: Modern Qt-based GUI framework
- **NumPy**: Numerical operations and data storage

---

## Prerequisites and Setup

### 1. Create Project Directory
```bash
mkdir face_recognition_app
cd face_recognition_app
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Create `requirements.txt`:
```
opencv-python>=4.8.0
mediapipe>=0.10.0
face-recognition>=1.3.0
numpy>=1.24.0
PySide6>=6.5.0
```

Install all packages:
```bash
pip install -r requirements.txt
```

**Important Notes:**
- `face-recognition` requires CMake and dlib (may take time to install)
- On Windows, you may need Visual Studio Build Tools
- On macOS, ensure you have Xcode Command Line Tools

---

## File Structure

Create this directory structure:
```
face_recognition_app/
├── run.py                 # Entry point script
├── requirements.txt       # Dependencies
├── data/                  # Database storage (created automatically)
│   ├── index.json        # Face metadata index
│   └── faces/            # Individual face folders
│       └── [person_name]/
│           ├── face.jpg       # Face image
│           └── embedding.npy  # 128D encoding
└── src/                   # Source code
    ├── camera_handler.py  # Webcam management
    ├── face_detector.py   # Detection & encoding
    ├── face_database.py   # Face storage & matching
    └── main.py           # GUI application
```

Create the `src` directory:
```bash
mkdir src
```

---

## Understanding the Architecture

### System Flow

```
┌─────────────────┐
│   run.py        │  Entry point
└────────┬────────┘
         │
         v
┌─────────────────┐
│   main.py       │  GUI & orchestration
└────┬───┬───┬────┘
     │   │   │
     v   v   v
┌────────┐ ┌────────────┐ ┌──────────────┐
│ camera │ │   face     │ │    face      │
│handler │ │  detector  │ │  database    │
└────────┘ └────────────┘ └──────────────┘
     │           │                │
     v           v                v
  OpenCV    MediaPipe +       Filesystem
            face_recognition   + JSON
```

### Component Responsibilities

**camera_handler.py**: Low-level webcam access
- Opens/closes camera device
- Reads frames (BGR images)
- Provides frame dimensions

**face_detector.py**: Face processing
- Detects face bounding boxes (face_recognition library)
- Detects 468 facial landmarks (MediaPipe)
- Generates 128D face encodings
- Drawing utilities (boxes, landmarks, cropping)

**face_database.py**: Persistent storage
- Saves face images and encodings to disk
- Loads embeddings for matching
- Searches for closest match using Euclidean distance
- Manages JSON index of all faces

**main.py**: GUI application
- PySide6/Qt-based interface
- Real-time video display
- Face recognition with visual feedback
- Database management UI

---

## Building the Project

---

## Step 1: Camera Handler

**File:** `src/camera_handler.py`

This module handles all webcam interactions using OpenCV's VideoCapture API.

### Step 1.1: Imports and Class Setup

```python
"""
Camera handler for live video capture.
"""
import cv2
import numpy as np
from typing import Optional


class CameraHandler:
    """Manages webcam video capture."""
    
    def __init__(self, camera_index: int = 0):
        # TODO: Store the camera_index parameter as an instance variable
        self.camera_index = camera_index
        
        # TODO: Initialize capture to None (will be set when camera starts)
        self.capture = None
        
        # TODO: Initialize running state flag to False
        self.is_running = False
```

**Fill-in hints:**
- First blank: instance variable name for camera index
- Second blank: what value should we assign?
- Third blank: initial capture object state
- Fourth blank: instance variable for tracking if camera is running
- Fifth blank: initial running state

### Step 1.2: Start Method

```python
    def start(self) -> bool:
        """
        Start the camera capture.
        
        Returns:
            bool: True if camera started successfully, False if failed
        """
        # If camera is already running, return success
        if self.capture is not None:
            return True
        
        # TODO: Initialize OpenCV VideoCapture with the camera_index
        self.capture = cv2.VideoCapture(self.camera_index)
        
        # TODO: Check if camera opened successfully using .isOpened() method
        if not self.capture.isOpened():
            # Failed - cleanup and return False
            self.capture = None
            return False
        
        # TODO: Set the is_running flag to True
        self.is_running = True
        return True
```

**Fill-in hints:**
- Check if capture object exists
- OpenCV VideoCapture constructor
- Method to check if camera device opened
- Update running state

### Step 1.3: Stop Method

```python
    def stop(self):
        """
        Stop the camera capture and release resources.
        """
        # TODO: Set is_running flag to False
        self.is_running = False
        
        # Release camera resources if active
        if self.capture is not None:
            # TODO: Call the release() method to free the camera
            self.capture.release()
            self.capture = None
```

**Fill-in hints:**
- Update running state to stopped
- Check if capture exists
- OpenCV method to release camera device

### Step 1.4: Read Frame Method

```python
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame from the camera.
        
        Returns:
            np.ndarray: BGR image (H x W x 3), or None if failed
        """
        # Check if camera is initialized and running
        if self.capture is None or not self.is_running:
            return None
        
        # TODO: Read frame using capture.read() method
        # This returns (success_flag, frame_array)
        ret, frame = self.capture.read()
        
        # Return frame if successful, otherwise None
        if ret:
            return frame
        return None
```

**Fill-in hints:**
- Check capture object and running state
- OpenCV method to read a frame (returns tuple)

### Step 1.5: Get Frame Size Method

```python
    def get_frame_size(self) -> tuple:
        """
        Get current frame dimensions.
        
        Returns:
            tuple: (width, height) in pixels
        """
        # Return default if not initialized
        if self.capture is None:
            return (640, 480)
        
        # TODO: Query camera properties using cv2.CAP_PROP_FRAME_WIDTH
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # TODO: Query camera properties using cv2.CAP_PROP_FRAME_HEIGHT
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return (width, height)
```

**Fill-in hints:**
- Check if capture is initialized
- OpenCV property constant for frame width
- OpenCV property constant for frame height

---

## Step 2: Face Detector

**File:** `src/face_detector.py`

This module handles face detection using two complementary systems:
- **face_recognition**: Fast face bounding box detection
- **MediaPipe**: Detailed 468-point facial landmarks

### Step 2.1: Imports and Initialization

```python
"""
Face detection and recognition module.
Uses MediaPipe for face detection/landmarks and face_recognition for embeddings.
"""
import cv2
import numpy as np
import mediapipe as mp
import face_recognition
from typing import List, Tuple, Optional


class FaceDetector:
    """Handles face detection, landmark detection, and face encoding."""
    
    def __init__(self):
        # TODO: Get MediaPipe solutions.face_mesh module
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # TODO: Create FaceMesh object with these parameters:
        # - static_image_mode=False (for video)
        # - max_num_faces=10
        # - min_detection_confidence=0.5
        # - min_tracking_confidence=0.5
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # TODO: Get MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
```

**Fill-in hints:**
- MediaPipe module for face mesh detection
- FaceMesh constructor
- Parameters for video vs static images, number of faces, and confidence thresholds
- MediaPipe drawing utilities modules

### Step 2.2: Detect Faces Method

```python
    def detect_faces(self, image: np.ndarray) -> Tuple[List, List]:
        """
        Detect faces and extract facial landmarks.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            Tuple of (face_locations, face_landmarks)
        """
        # TODO: Convert BGR to RGB (required by face detection libraries)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # TODO: Get face locations using face_recognition.face_locations()
        # Use model="hog" for speed (or "cnn" for accuracy)
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        
        # TODO: Get facial landmarks using MediaPipe face_mesh.process()
        results = self.face_mesh.process(rgb_image)
        
        return face_locations, results
```

**Fill-in hints:**
- OpenCV color conversion code from BGR to RGB
- face_recognition function for detecting face locations
- Model parameter (hog is faster, cnn is more accurate)
- MediaPipe method to process an image

### Step 2.3: Get Face Encoding Method

```python
    def get_face_encoding(self, image: np.ndarray, face_location: Tuple) -> Optional[np.ndarray]:
        """
        Generate 128-dimensional face encoding.
        
        Args:
            image: BGR image from OpenCV
            face_location: (top, right, bottom, left) tuple
            
        Returns:
            np.ndarray: 128D encoding, or None if failed
        """
        # TODO: Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # TODO: Generate encoding using face_recognition.face_encodings()
        # Pass the rgb_image and a list containing the face_location
        encodings = face_recognition.face_encodings(rgb_image, [face_location])
        
        # Return first encoding if successful
        if encodings:
            return encodings[0]
        return None
```

**Fill-in hints:**
- Color conversion method and constant
- face_recognition function to generate encodings
- face_location parameter (must be in a list)

### Step 2.4: Draw Face Boxes Method

```python
    def draw_face_boxes(self, image: np.ndarray, face_locations: List) -> np.ndarray:
        """
        Draw bounding boxes around detected faces.
        
        Args:
            image: BGR image
            face_locations: List of (top, right, bottom, left) tuples
            
        Returns:
            np.ndarray: Image with drawn boxes
        """
        # TODO: Create a copy to avoid modifying original
        output = image.copy()
        
        # Draw rectangle for each face
        for (top, right, bottom, left) in face_locations:
            # TODO: Draw rectangle using cv2.rectangle()
            # Parameters: image, (x1, y1), (x2, y2), color_BGR, thickness
            # Use green color (0, 255, 0) and thickness 2
            cv2.rectangle(output, (left, top), (right, bottom), 
                       (0, 255, 0), 2)
        
        return output
```

**Fill-in hints:**
- NumPy method to create a copy of an array
- OpenCV function to draw rectangles
- Corner coordinates (remember: OpenCV uses x,y not row,col)
- Rectangle parameters: top-left corner and bottom-right corner

### Step 2.5: Draw Face Landmarks Method

```python
    def draw_face_landmarks(self, image: np.ndarray, face_mesh_results) -> np.ndarray:
        """
        Draw MediaPipe facial landmarks (468 points).
        
        Args:
            image: BGR image
            face_mesh_results: MediaPipe face mesh results
            
        Returns:
            np.ndarray: Image with drawn landmarks
        """
        # TODO: Create a copy
        output = image.copy()
        
        # Check if faces were detected
        if face_mesh_results.multi_face_landmarks:
            # Process each detected face
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                # TODO: Draw tessellation (full mesh)
                self.mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # TODO: Draw contours (emphasize features)
                self.mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        
        return output
```

**Fill-in hints:**
- Copy method
- MediaPipe results attribute for multiple face landmarks
- MediaPipe drawing method name
- Landmark list to draw

### Step 2.6: Crop Face Method

```python
    def crop_face(self, image: np.ndarray, face_location: Tuple, padding: int = 20) -> np.ndarray:
        """
        Crop face region with padding.
        
        Args:
            image: BGR image
            face_location: (top, right, bottom, left) tuple
            padding: Extra pixels around face
            
        Returns:
            np.ndarray: Cropped face image
        """
        top, right, bottom, left = face_location
        
        # Add padding but stay within image bounds
        height, width = image.shape[:2]
        
        # TODO: Add padding to top, ensuring >= 0
        top = max(0, top - padding)
        
        # TODO: Add padding to bottom, ensuring <= height
        bottom = min(height, bottom + padding)
        
        # TODO: Add padding to left, ensuring >= 0
        left = max(0, left - padding)
        
        # TODO: Add padding to right, ensuring <= width
        right = min(width, right + padding)
        
        # TODO: Crop using NumPy array slicing [y1:y2, x1:x2]
        return image[top:bottom, left:right]
```

**Fill-in hints:**
- Padding value for each side
- Image dimensions (height, width)
- NumPy slicing syntax for 2D arrays (remember: [rows, columns])

### Step 2.7: Cleanup Method

```python
    def cleanup(self):
        """Release MediaPipe resources."""
        # TODO: Close the face_mesh object
        self.face_mesh.close()
```

**Fill-in hints:**
- MediaPipe method to release resources

---

## Step 3: Face Database

**File:** `src/face_database.py`

This module manages persistent face storage using the filesystem and JSON index.

### Step 3.1: Imports and Initialization

```python
"""
Face database manager.
Handles saving, loading, and searching saved faces with embeddings.
"""
import os
import json
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class FaceDatabase:
    """Manages saved faces and their embeddings on the filesystem."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the face database.
        
        Args:
            data_dir: Root directory for storing face data
        """
        # TODO: Create Path object for data directory
        self.data_dir = Path(data_dir)
        
        # TODO: Create path for faces subdirectory
        self.faces_dir = self.data_dir / "faces"
        
        # TODO: Create path for index.json file
        self.index_file = self.data_dir / "index.json"
        
        # TODO: Create directories if they don't exist
        # Use mkdir with parents=True and exist_ok=True
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        
        # TODO: Initialize empty index dictionary
        self.index = {}
        
        # Load existing index from disk
        self._load_index()
```

**Fill-in hints:**
- Path constructor from pathlib
- Subdirectory names
- mkdir method parameters
- Empty dictionary

### Step 3.2: Load Index Method

```python
    def _load_index(self):
        """Load the index.json file into memory."""
        # TODO: Check if index_file exists using .exists() method
        if self.index_file.exists():
            # Load existing index
            # TODO: Open file in read mode and load JSON
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            # Create new empty index
            self.index = {}
            self._save_index()
```

**Fill-in hints:**
- Path method to check existence
- File mode for reading
- JSON method to load from file

### Step 3.3: Save Index Method

```python
    def _save_index(self):
        """Save the in-memory index to index.json."""
        # TODO: Open file in write mode
        with open(self.index_file, 'w') as f:
            # TODO: Write index as JSON with indentation for readability
            json.dump(self.index, f, indent=2)
```

**Fill-in hints:**
- File mode for writing
- JSON method to write to file
- Parameter for pretty-printing (use indent=2)

### Step 3.4: Sanitize Name Method (Helper)

```python
    @staticmethod
    def _sanitize_name(name: str) -> str:
        """
        Convert display name to filesystem-safe name.
        
        Examples: "John Doe" -> "john_doe"
        """
        # TODO: Replace non-alphanumeric characters with underscores
        # Use a list comprehension: for each character c, keep if alphanumeric, else use "_"
        safe = "".join(c if c.isalnum() else "_" for c in name)
        
        # Remove consecutive underscores
        while "__" in safe:
            safe = safe.replace("__", "_")
        
        # TODO: Strip underscores from edges and convert to lowercase
        return safe.strip("_").lower()
```

**Fill-in hints:**
- String method to check if character is alphanumeric
- String method to remove characters from edges
- String method to convert to lowercase

### Step 3.5: Save Face Method

```python
    def save_face(self, name: str, face_image: np.ndarray, embedding: np.ndarray) -> bool:
        """
        Save a face image and its embedding.
        
        Args:
            name: Person's name (will be sanitized)
            face_image: Cropped face image (BGR)
            embedding: 128D face encoding
            
        Returns:
            bool: True if saved successfully
        """
        # TODO: Sanitize name for filesystem
        safe_name = self._sanitize_name(name)
        
        # TODO: Create directory path for this person
        person_dir = self.faces_dir / safe_name
        
        # TODO: Create directory
        person_dir.mkdir(exist_ok=True)
        
        # TODO: Save face image as JPEG using cv2.imwrite()
        face_path = person_dir / "face.jpg"
        cv2.imwrite(str(face_path), face_image)
        
        # TODO: Save embedding as numpy binary file using np.save()
        embedding_path = person_dir / "embedding.npy"
        np.save(str(embedding_path), embedding)
        
        # TODO: Update index with metadata dictionary
        self.index[safe_name] = {
            "display_name": name,  # Original name
            "face_path": str(face_path),
            "embedding_path": str(embedding_path)
        }
        
        # Persist index to disk
        self._save_index()
        return True
```

**Fill-in hints:**
- Static method to sanitize the name
- safe_name for directory
- mkdir method
- OpenCV function to write image
- NumPy function to save array
- Dictionary keys for display name, face path, embedding path

### Step 3.6: Get All Faces Method

```python
    def get_all_faces(self) -> List[Dict]:
        """
        Get all saved faces with their information.
        
        Returns:
            List[Dict]: List of face dictionaries
        """
        faces = []
        
        # TODO: Iterate over index items (use .items() method)
        for safe_name, data in self.index.items():
            faces.append({
                "name": safe_name,
                # TODO: Get display_name from data, default to safe_name if missing
                "display_name": data.get("display_name", safe_name),
                "face_path": data["face_path"],
                "embedding_path": data["embedding_path"]
            })
        
        return faces
```

**Fill-in hints:**
- Dictionary method to get key-value pairs
- Dictionary key for display name

### Step 3.7: Load Embedding Method

```python
    def load_embedding(self, name: str) -> Optional[np.ndarray]:
        """
        Load embedding for a specific person.
        
        Args:
            name: Person's safe name
            
        Returns:
            np.ndarray: 128D embedding, or None if not found
        """
        # TODO: Check if name exists in index
        if name in self.index:
            embedding_path = self.index[name]["embedding_path"]
            
            # TODO: Check if file exists using os.path.exists()
            if os.path.exists(embedding_path):
                # TODO: Load numpy array using np.load()
                return np.load(embedding_path)
        
        return None
```

**Fill-in hints:**
- Check dictionary membership
- os.path function to check file existence
- NumPy function to load array from file

### Step 3.8: Load All Embeddings Method

```python
    def load_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load all saved embeddings into memory.
        
        Returns:
            Dict mapping safe_name to 128D embedding
        """
        embeddings = {}
        
        # TODO: Iterate over index keys
        for safe_name in self.index.keys():
            embedding = self.load_embedding(safe_name)
            if embedding is not None:
                embeddings[safe_name] = embedding
        
        return embeddings
```

**Fill-in hints:**
- Dictionary method to get keys

### Step 3.9: Find Closest Match Method

```python
    def find_closest_match(self, query_embedding: np.ndarray, 
                          threshold: float = 0.6) -> Optional[Tuple[str, float]]:
        """
        Find the closest matching face in the database.
        
        Uses Euclidean distance. Typical values:
        - Same person: 0.0 - 0.4
        - Different people: 0.6 - 1.2
        
        Args:
            query_embedding: 128D encoding to match
            threshold: Maximum distance to consider a match (default: 0.6)
            
        Returns:
            Tuple of (display_name, distance) if match found, else None
        """
        # Load all embeddings
        embeddings = self.load_all_embeddings()
        
        if not embeddings:
            return None
        
        # Initialize tracking variables
        best_match = None
        best_distance = float('inf')  # Start with infinity
        
        # TODO: Compare query against each saved face
        for name, saved_embedding in embeddings.items():
            # TODO: Calculate Euclidean distance using np.linalg.norm()
            # Distance = norm(query_embedding - saved_embedding)
            distance = np.linalg.norm(query_embedding - saved_embedding)
            
            # TODO: Update best match if this distance is lower
            if distance < best_distance:
                best_distance = distance
                best_match = name
        
        # TODO: Only return match if within threshold
        if best_distance <= threshold:
            # Get display name
            display_name = self.index[best_match].get("display_name", best_match)
            return (display_name, best_distance)
        
        return None
```

**Fill-in hints:**
- String for infinity in Python
- Dictionary method for key-value pairs
- NumPy function to calculate vector norm
- Variable to compare against
- Threshold parameter

### Step 3.10: Rename Face Method

```python
    def rename_face(self, old_name: str, new_name: str) -> bool:
        """
        Rename a saved face (display name only).
        
        Args:
            old_name: Current safe name
            new_name: New display name
            
        Returns:
            bool: True if successful
        """
        # TODO: Check if old_name exists in index
        if old_name not in self.index:
            return False
        
        # TODO: Update display_name in index
        self.index[old_name]["display_name"] = new_name
        
        self._save_index()
        return True
```

**Fill-in hints:**
- Check dictionary membership
- Dictionary key to update

### Step 3.11: Delete Face Method

```python
    def delete_face(self, name: str) -> bool:
        """
        Delete a saved face and its data.
        
        Args:
            name: Safe name of person to delete
            
        Returns:
            bool: True if successful
        """
        # TODO: Check if name exists
        if name not in self.index:
            return False
        
        # Remove from filesystem
        person_dir = self.faces_dir / name
        if person_dir.exists():
            import shutil
            # TODO: Recursively delete directory using shutil.rmtree()
            shutil.rmtree(person_dir)
        
        # TODO: Remove from index using del keyword
        del self.index[name]
        
        self._save_index()
        return True
```

**Fill-in hints:**
- Check dictionary membership
- shutil function to remove directory tree
- Python keyword to delete dictionary entry

### Step 3.12: Search Faces Method

```python
    def search_faces(self, query: str) -> List[Dict]:
        """
        Search for faces by name (case-insensitive substring match).
        
        Args:
            query: Search string
            
        Returns:
            List[Dict]: Matching face dictionaries
        """
        # TODO: Convert query to lowercase
        query_lower = query.lower()
        results = []
        
        # Search through all faces
        for face in self.get_all_faces():
            # TODO: Check if query is substring of display_name (case-insensitive)
            if query_lower in face["display_name"].lower():
                results.append(face)
        
        return results
```

**Fill-in hints:**
- String method to convert to lowercase
- Check substring membership (after lowercasing)

---

## Step 4: Main GUI Application

**File:** `src/main.py`

This is the largest file - it creates the Qt GUI and orchestrates all components.

### Step 4.1: Imports

```python
"""
Main GUI application for face recognition.
Built with PySide6 (Qt for Python).
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# TODO: Import Qt widgets - fill in the missing widget names
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QScrollArea, QFileDialog,
    QStatusBar, QToolBar, QMessageBox, QInputDialog, QFrame
)

# TODO: Import Qt core classes
from PySide6.QtCore import Qt, QTimer, Signal, QSize

# TODO: Import Qt GUI classes
from PySide6.QtGui import QImage, QPixmap, QAction

# TODO: Import our custom modules
from face_detector import FaceDetector
from face_database import FaceDatabase
from camera_handler import CameraHandler
```

**Fill-in hints:**
- The three class names from our custom modules

### Step 4.2: FaceItemWidget Class Setup

```python
class FaceItemWidget(QFrame):
    """Widget representing a single saved face in the list."""
    
    # TODO: Define a signal that emits two strings (old_name, new_name)
    rename_requested = Signal(str, str)
    
    def __init__(self, name: str, display_name: str, face_path: str):
        super().__init__()
        
        # TODO: Store parameters as instance variables
        self.name = name
        self.display_name = display_name
        self.face_path = face_path
        
        # Create horizontal layout
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
```

**Fill-in hints:**
- Qt Signal class
- Instance variable names for the three parameters
- Qt horizontal layout class

### Step 4.3: FaceItemWidget Layout (continued)

```python
        # TODO: Create thumbnail label with fixed size 64x64
        self.thumbnail = QLabel()
        self.thumbnail.setFixedSize(64, 64)
        self.thumbnail.setScaledContents(True)
        self.load_thumbnail()
        layout.addWidget(self.thumbnail)
        
        # TODO: Create name label with display_name
        self.name_label = QLabel(display_name)
        self.name_label.setStyleSheet("font-size: 12pt;")
        layout.addWidget(self.name_label, 1)  # Stretch factor
        
        # TODO: Create rename button with emoji pencil "✏️"
        rename_btn = QPushButton("✏️")
        rename_btn.setFixedSize(30, 30)
        rename_btn.clicked.connect(self.on_rename_clicked)
        layout.addWidget(rename_btn)
        
        self.setLayout(layout)
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(1)
```

**Fill-in hints:**
- QLabel constructor
- Fixed size dimensions
- QPushButton constructor

### Step 4.4: FaceItemWidget Load Thumbnail

```python
    def load_thumbnail(self):
        """Load and display face thumbnail."""
        # TODO: Check if face_path exists using Path().exists()
        if Path(self.face_path).exists():
            # TODO: Load image using cv2.imread()
            image = cv2.imread(self.face_path)
            
            if image is not None:
                # TODO: Resize to 64x64 using cv2.resize()
                image = cv2.resize(image, (64, 64))
                
                # TODO: Convert BGR to RGB for Qt
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                
                # TODO: Create QImage from numpy array data
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, 
                                  QImage.Format_RGB888)
                
                # TODO: Set pixmap on thumbnail label
                self.thumbnail.setPixmap(QPixmap.fromImage(qt_image))
```

**Fill-in hints:**
- Path constructor and exists method
- OpenCV read, resize, and color conversion functions
- QImage constructor
- QPixmap static method

### Step 4.5: FaceItemWidget Rename Handler

```python
    def on_rename_clicked(self):
        """Handle rename button click."""
        # TODO: Show input dialog using QInputDialog.getText()
        # Parameters: parent, title, label, text=default_value
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Face",
            "Enter new name:",
            text=self.display_name
        )
        
        # TODO: Check if user clicked OK and name is different
        if ok and new_name and new_name != self.display_name:
            # TODO: Emit the rename_requested signal
            self.rename_requested.emit(self.name, new_name)
            
            # Update label
            self.name_label.setText(new_name)
            self.display_name = new_name
```

**Fill-in hints:**
- QInputDialog class and method for text input
- Current display name
- Signal to emit

### Step 4.6: FaceRecognitionApp Class Setup

```python
class FaceRecognitionApp(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # TODO: Initialize core components
        self.detector = FaceDetector()
        self.database = FaceDatabase("data")
        self.camera = CameraHandler()
        
        # TODO: Initialize state variables
        self.current_mode = None  # 'live', 'image', or None
        self.current_image = None
        self.current_face_locations = []
        self.current_face_landmarks = None
        
        # TODO: Create QTimer for live feed updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live_feed)
        
        # Setup UI and load faces
        self.setup_ui()
        self.load_saved_faces()
```

**Fill-in hints:**
- Three class constructors for our components
- QTimer constructor

### Step 4.7: Setup UI - Window and Main Layout

```python
    def setup_ui(self):
        """Initialize the user interface."""
        
        # TODO: Set window title
        self.setWindowTitle("Face Recognition App")
        
        # TODO: Set window geometry (x, y, width, height)
        self.setGeometry(100, 100, 1200, 800)
        
        # TODO: Create and set central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # TODO: Create main horizontal layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # TODO: Create left vertical layout for display area
        left_layout = QVBoxLayout()
```

**Fill-in hints:**
- Window title text
- QWidget constructor
- QHBoxLayout for main layout
- QVBoxLayout for left side

### Step 4.8: Setup UI - Toolbar

```python
        # Create toolbar
        toolbar = QToolBar()
        
        # TODO: Create Live Feed button with emoji "📹"
        self.live_btn = QPushButton("📹 Live Feed")
        self.live_btn.clicked.connect(self.start_live_feed)
        toolbar.addWidget(self.live_btn)
        
        # TODO: Create Upload button with emoji "📁"
        self.upload_btn = QPushButton("📁 Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        toolbar.addWidget(self.upload_btn)
        
        # TODO: Create Save Face button with emoji "💾"
        self.save_face_btn = QPushButton("💾 Save Face")
        self.save_face_btn.clicked.connect(self.save_current_face)
        self.save_face_btn.setEnabled(False)  # Disabled initially
        toolbar.addWidget(self.save_face_btn)
        
        left_layout.addWidget(toolbar)
```

**Fill-in hints:**
- QPushButton constructor for each button

### Step 4.9: Setup UI - Display Label

```python
        # TODO: Create display label for video/image
        self.display_label = QLabel()
        self.display_label.setMinimumSize(800, 600)
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #2b2b2b; color: white;")
        self.display_label.setText("Click 'Live Feed' or 'Upload Image' to start")
        left_layout.addWidget(self.display_label)
        
        # TODO: Add left_layout to main_layout with stretch factor 3
        main_layout.addLayout(left_layout, 3)
```

**Fill-in hints:**
- QLabel constructor
- Qt alignment constant for center
- Stretch factor (3 = 75% of width)

### Step 4.10: Setup UI - Right Side Panel

```python
        # TODO: Create right vertical layout
        right_layout = QVBoxLayout()
        
        # Search label
        search_label = QLabel("Search Faces:")
        right_layout.addWidget(search_label)
        
        # TODO: Create search input field
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type to search...")
        self.search_input.textChanged.connect(self.on_search_changed)
        right_layout.addWidget(self.search_input)
        
        # TODO: Create scroll area for faces list
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumWidth(300)
        
        # TODO: Create container widget for faces
        self.faces_container = QWidget()
        self.faces_layout = QVBoxLayout()
        self.faces_layout.setAlignment(Qt.AlignTop)
        self.faces_container.setLayout(self.faces_layout)
        
        self.scroll_area.setWidget(self.faces_container)
        right_layout.addWidget(self.scroll_area)
        
        # TODO: Add right_layout to main_layout with stretch factor 1
        main_layout.addLayout(right_layout, 1)
        
        # TODO: Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
```

**Fill-in hints:**
- QVBoxLayout for right side
- QLineEdit for search
- QScrollArea for scrollable list
- QWidget for container
- QVBoxLayout for faces list
- Stretch factor for right side (1 = 25%)
- QStatusBar constructor

### Step 4.11: Start Live Feed Method

```python
    def start_live_feed(self):
        """Start or stop live camera feed."""
        
        # TODO: Check if currently in live mode
        if self.current_mode == 'live':
            # Stop camera
            self.timer.stop()
            self.camera.stop()
            self.current_mode = None
            self.live_btn.setText("📹 Live Feed")
            self.display_label.setText("Live feed stopped")
            self.save_face_btn.setEnabled(False)
            self.status_bar.showMessage("Live feed stopped")
            return
        
        # TODO: Try to start camera
        if self.camera.start():
            self.current_mode = 'live'
            
            # TODO: Start timer with 30ms interval (~30 FPS)
            self.timer.start(30)
            
            self.live_btn.setText("⏹️ Stop Feed")
            self.status_bar.showMessage("Live feed started")
        else:
            # TODO: Show warning message box
            QMessageBox.warning(self, "Camera Error", "Failed to start camera")
```

**Fill-in hints:**
- Mode string for live camera
- Camera method to start
- Timer interval in milliseconds
- QMessageBox class for warning dialog

### Step 4.12: Update Live Feed Method

```python
    def update_live_feed(self):
        """Update live feed frame (called by timer)."""
        
        # TODO: Read frame from camera
        frame = self.camera.read_frame()
        if frame is None:
            return
        
        # TODO: Store copy as current_image
        self.current_image = frame.copy()
        
        # TODO: Detect faces and landmarks
        self.current_face_locations, self.current_face_landmarks = \
            self.detector.detect_faces(frame)
        
        # TODO: Draw face boxes if faces detected
        if self.current_face_locations:
            frame = self.detector.draw_face_boxes(frame, self.current_face_locations)
            self.save_face_btn.setEnabled(True)
        else:
            self.save_face_btn.setEnabled(False)
        
        # TODO: Draw landmarks if detected
        if self.current_face_landmarks:
            frame = self.detector.draw_face_landmarks(frame, self.current_face_landmarks)
        
        # TODO: Try to match first face
        if self.current_face_locations:
            first_face = self.current_face_locations[0]
            
            # TODO: Get face encoding
            encoding = self.detector.get_face_encoding(self.current_image, first_face)
            
            if encoding is not None:
                # TODO: Find closest match in database
                match = self.database.find_closest_match(encoding)
                
                if match:
                    name, distance = match
                    self.status_bar.showMessage(f"Match: {name} (distance: {distance:.3f})")
                else:
                    self.status_bar.showMessage("No match found")
        
        # TODO: Display frame
        self.display_frame(frame)
```

**Fill-in hints:**
- Camera method to read frame
- NumPy copy method
- Detector methods for face detection, drawing boxes, and landmarks
- Enable/disable states (boolean)
- Detector method to get encoding
- Database method to find match
- Method to display frame (defined later)

### Step 4.13: Upload Image Method

```python
    def upload_image(self):
        """Upload and process a static image."""
        
        # TODO: Open file dialog using QFileDialog.getOpenFileName()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if not file_path:
            return  # User cancelled
        
        # Stop live feed if running
        if self.current_mode == 'live':
            self.timer.stop()
            self.camera.stop()
            self.live_btn.setText("📹 Live Feed")
        
        self.current_mode = 'image'
        
        # TODO: Load image using cv2.imread()
        image = cv2.imread(file_path)
        if image is None:
            QMessageBox.warning(self, "Error", "Failed to load image")
            return
        
        # TODO: Store copy and detect faces
        self.current_image = image.copy()
        self.current_face_locations, self.current_face_landmarks = \
            self.detector.detect_faces(image)
        
        # TODO: Draw boxes if faces found
        if self.current_face_locations:
            image = self.detector.draw_face_boxes(image, self.current_face_locations)
            self.save_face_btn.setEnabled(True)
        else:
            self.save_face_btn.setEnabled(False)
            QMessageBox.information(self, "No Faces", "No faces detected in image")
        
        # TODO: Draw landmarks
        if self.current_face_landmarks:
            image = self.detector.draw_face_landmarks(image, self.current_face_landmarks)
        
        # TODO: Try to match first face
        if self.current_face_locations:
            first_face = self.current_face_locations[0]
            encoding = self.detector.get_face_encoding(self.current_image, first_face)
            
            if encoding is not None:
                match = self.database.find_closest_match(encoding)
                if match:
                    name, distance = match
                    self.status_bar.showMessage(f"Match: {name} (distance: {distance:.3f})")
                else:
                    self.status_bar.showMessage("No match found")
        
        # Display image
        self.display_frame(image)
```

**Fill-in hints:**
- QFileDialog class and method
- Mode string for image
- OpenCV imread function
- NumPy copy method
- Detector methods
- QMessageBox information dialog
- Detector and database methods

### Step 4.14: Save Current Face Method

```python
    def save_current_face(self):
        """Save the first detected face to the database."""
        
        # TODO: Validate we have a face
        if not self.current_face_locations or self.current_image is None:
            QMessageBox.warning(self, "No Face", "No face detected to save")
            return
        
        # TODO: Prompt user for name using QInputDialog.getText()
        name, ok = QInputDialog.getText(self, "Save Face", "Enter person's name:")
        if not ok or not name:
            return
        
        # Get first face
        first_face = self.current_face_locations[0]
        
        # TODO: Crop face using detector
        face_image = self.detector.crop_face(self.current_image, first_face)
        
        # TODO: Generate encoding
        encoding = self.detector.get_face_encoding(self.current_image, first_face)
        if encoding is None:
            QMessageBox.warning(self, "Error", "Failed to generate face encoding")
            return
        
        # TODO: Save to database
        if self.database.save_face(name, face_image, encoding):
            QMessageBox.information(self, "Success", f"Face saved as '{name}'")
            self.load_saved_faces()  # Refresh list
        else:
            QMessageBox.warning(self, "Error", "Failed to save face")
```

**Fill-in hints:**
- QInputDialog class and method
- Detector method to crop face
- Detector method to get encoding
- Database method to save face

### Step 4.15: Load Saved Faces Method

```python
    def load_saved_faces(self, search_query: str = ""):
        """Load and display saved faces in the sidebar."""
        
        # TODO: Clear existing widgets (iterate in reverse)
        for i in reversed(range(self.faces_layout.count())):
            widget = self.faces_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # TODO: Get faces from database (filtered or all)
        if search_query:
            faces = self.database.search_faces(search_query)
        else:
            faces = self.database.get_all_faces()
        
        # TODO: Create widget for each face
        for face in faces:
            widget = FaceItemWidget(
                face['name'],
                face['display_name'],
                face['face_path']
            )
            # TODO: Connect rename signal
            widget.rename_requested.connect(self.on_face_renamed)
            self.faces_layout.addWidget(widget)
```

**Fill-in hints:**
- Database methods for search and get all
- Dictionary keys: name, display_name, face_path

### Step 4.16: Search Changed Handler

```python
    def on_search_changed(self, text: str):
        """Handle search input changes."""
        # TODO: Reload faces with filter
        self.load_saved_faces(text)
```

**Fill-in hints:**
- Method to reload faces with search query

### Step 4.17: Face Renamed Handler

```python
    def on_face_renamed(self, old_name: str, new_name: str):
        """Handle face rename request."""
        # TODO: Rename in database
        if self.database.rename_face(old_name, new_name):
            self.status_bar.showMessage(f"Renamed to '{new_name}'")
```

**Fill-in hints:**
- Database method to rename face

### Step 4.18: Display Frame Method

```python
    def display_frame(self, frame: np.ndarray):
        """Display a frame in the GUI."""
        
        # TODO: Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Calculate resize to fit display
        h, w, ch = rgb_frame.shape
        display_width = self.display_label.width()
        display_height = self.display_label.height()
        
        # Calculate aspect ratio
        aspect = w / h
        if display_width / display_height > aspect:
            new_height = display_height
            new_width = int(new_height * aspect)
        else:
            new_width = display_width
            new_height = int(new_width / aspect)
        
        # TODO: Resize frame using cv2.resize()
        resized = cv2.resize(rgb_frame, (new_width, new_height))
        h, w, ch = resized.shape
        
        # TODO: Convert to Qt image
        bytes_per_line = ch * w
        qt_image = QImage(resized.data, w, h, bytes_per_line, 
                          QImage.Format_RGB888)
        
        # TODO: Set pixmap on display label
        self.display_label.setPixmap(QPixmap.fromImage(qt_image))
```

**Fill-in hints:**
- OpenCV color conversion function and constant
- OpenCV resize function
- QImage constructor
- QPixmap static method

### Step 4.19: Close Event Handler

```python
    def closeEvent(self, event):
        """Handle window close event - cleanup resources."""
        # TODO: Stop timer
        self.timer.stop()
        
        # TODO: Stop camera
        self.camera.stop()
        
        # TODO: Cleanup detector
        self.detector.cleanup()
        
        # TODO: Accept event to allow close
        event.accept()
```

**Fill-in hints:**
- Timer method to stop
- Camera method to stop
- Detector cleanup method
- Event method to accept

### Step 4.20: Main Entry Point

```python
def main():
    """Main entry point for the application."""
    
    # TODO: Create Qt application
    app = QApplication(sys.argv)
    
    # TODO: Create main window
    window = FaceRecognitionApp()
    
    # TODO: Show window
    window.show()
    
    # TODO: Start event loop and exit with return code
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

**Fill-in hints:**
- QApplication constructor
- FaceRecognitionApp constructor
- Window method to show
- Application method to start event loop

---

## Step 5: Run Script

**File:** `run.py` (in project root, not in src/)

This is the entry point that users will execute.

```python
#!/usr/bin/env python3
"""
Launcher script for Face Recognition App.
Run this from the project root directory.
"""
import sys
import os

# TODO: Add src directory to Python path
# Join the directory containing this file with 'src'
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

# TODO: Import main function from main module (main.py in src/)
from main import main

if __name__ == "__main__":
    # Display startup message
    print("Starting Face Recognition App...")
    print("Press Ctrl+C to exit")
    
    # TODO: Launch the main application
    main()
```

**Fill-in hints:**
- Subdirectory name
- Function name to import
- Function to call

---

## Testing Your Application

### 1. Verify Installation
First, test that all dependencies are installed correctly:

```python
# Create test_installation.py
import cv2
import mediapipe
import face_recognition
import numpy
from PySide6 import QtWidgets

print("✓ All dependencies installed successfully!")
print(f"  OpenCV: {cv2.__version__}")
print(f"  MediaPipe: {mediapipe.__version__}")
print(f"  NumPy: {numpy.__version__}")
```

Run: `python test_installation.py`

### 2. Run the Application

```bash
python run.py
```

### 3. Test Features

**Test 1: Live Feed**
1. Click "📹 Live Feed" button
2. You should see your webcam feed
3. Green boxes should appear around detected faces
4. Facial landmarks (mesh) should overlay on faces

**Test 2: Save a Face**
1. With live feed running and face detected
2. Click "💾 Save Face" button
3. Enter your name in the dialog
4. Check that your face appears in the right sidebar

**Test 3: Face Recognition**
1. Move away from camera and return
2. Status bar should show "Match: [Your Name] (distance: X.XXX)"
3. Distance should be < 0.4 for same person

**Test 4: Upload Image**
1. Click "📁 Upload Image"
2. Select a photo containing a face
3. Face should be detected and matched

**Test 5: Search**
1. Save multiple faces with different names
2. Type in the search box
3. Face list should filter in real-time

### 4. Troubleshooting

**Camera won't open:**
- Check if another application is using the camera
- Try changing `camera_index` to 1 or 2 in `camera_handler.py`

**No faces detected:**
- Ensure good lighting
- Face the camera directly
- Try moving closer/further

**Poor recognition accuracy:**
- Adjust `threshold` in `find_closest_match()` (default: 0.6)
- Lower threshold = stricter matching
- Save multiple images of the same person

**Slow performance:**
- In `face_detector.py`, use `model="hog"` instead of `"cnn"`
- Reduce `max_num_faces` in MediaPipe initialization
- Lower camera resolution

---

## Congratulations!

You've built a complete face recognition application with:
- Real-time video processing
- Deep learning-based face detection
- Face encoding and matching
- Persistent database storage
- Professional Qt GUI
- Search and management features

### Next Steps to Enhance Your Project:

1. **Add delete functionality**: Add delete buttons to FaceItemWidget
2. **Multiple photos per person**: Store multiple embeddings and average them
3. **Confidence display**: Show match confidence visually (color-coded)
4. **Face detection settings**: Add UI controls for detection parameters
5. **Export/Import**: Add database backup and restore features
6. **Webcam selection**: Let users choose which camera to use
7. **Frame rate display**: Show FPS counter on live feed
8. **Batch upload**: Process multiple images at once

---

## Understanding Key Concepts

### Face Encoding (128D Vector)
- Deep neural network converts face to 128 numbers
- Similar faces → similar vectors
- Different faces → different vectors
- Enables fast mathematical comparison

### Euclidean Distance
- Measures similarity between two face encodings
- Formula: √(Σ(a_i - b_i)²)
- Smaller distance = more similar faces
- Threshold (0.6) separates matches from non-matches

### MediaPipe Face Mesh
- Detects 468 3D facial landmarks
- Includes eyes, nose, mouth, face contour
- Uses machine learning models
- Real-time performance (~30 FPS)

### Qt Signal/Slot Mechanism
- Signals: Events emitted by widgets
- Slots: Functions that respond to signals
- Loose coupling between components
- Example: button.clicked → slot function

### File Structure Best Practices
- `src/`: All source code modules
- `data/`: Runtime-generated data (gitignore this)
- `run.py`: Entry point at project root
- `requirements.txt`: Dependency specification

---

## Additional Resources

**Libraries Documentation:**
- OpenCV: https://docs.opencv.org/
- MediaPipe: https://google.github.io/mediapipe/
- face_recognition: https://face-recognition.readthedocs.io/
- PySide6: https://doc.qt.io/qtforpython/

**Computer Vision Concepts:**
- Face Detection vs Recognition
- Haar Cascades vs Deep Learning
- HOG (Histogram of Oriented Gradients)
- FaceNet and embeddings

**Python Best Practices:**
- Type hints for better code clarity
- Docstrings for documentation
- Error handling with try/except
- Virtual environments for dependencies

---

**End of Tutorial**

Remember: The blanks in this tutorial are designed to make you think about:
- What data needs to be stored (instance variables)
- What functions/methods to call
- How components connect together
- The flow of data through the application

By filling in the blanks yourself, you'll gain a deeper understanding of how each piece works rather than just copying complete code. Good luck!
