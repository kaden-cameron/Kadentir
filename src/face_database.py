"""
Face database manager.
Handles saving, loading, and searching saved faces with embeddings.
"""
import os
import json
import numpy as np
import cv2
import shutil
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class FaceDatabase:
    """Manages saved faces and their embeddings on the filesystem."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.faces_dir = self.data_dir / "faces"
        self.index_file = self.data_dir / "index.json"
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self._load_index()

    def _load_index(self):
        """Load the index.json file into memory."""
        if self.index_file.exists():
            with open(self.index_file, "r", encoding="utf-8") as f:
                self.index = json.load(f)
        else:
            self.index = {}
            self._save_index()

    def _save_index(self):
        """Save the in-memory index to index.json."""
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Convert display name to filesystem-safe name. E.g. 'John Doe' -> 'john_doe'."""
        safe = "".join(c if c.isalnum() else "_" for c in name)
        while "__" in safe:
            safe = safe.replace("__", "_")
        return safe.strip("_").lower()

    def save_face(self, name: str, face_image: np.ndarray, embedding: np.ndarray) -> bool:
        """
        Save a face image and its embedding.
        name: Person's name (will be sanitized)
        face_image: Cropped face image (BGR)
        embedding: 128D face encoding
        Returns True if saved successfully.
        """
        safe_name = self._sanitize_name(name)
        if not safe_name:
            return False
        person_dir = self.faces_dir / safe_name
        person_dir.mkdir(exist_ok=True)
        face_path = person_dir / "face.jpg"
        cv2.imwrite(str(face_path), face_image)
        embedding_path = person_dir / "embedding.npy"
        np.save(str(embedding_path), embedding.astype(np.float64))
        self.index[safe_name] = {
            "display_name": name,
            "face_path": str(face_path),
            "embedding_path": str(embedding_path),
        }
        self._save_index()
        return True

    def get_all_faces(self) -> List[Dict]:
        """Get all saved faces with their information."""
        faces = []
        for safe_name, data in self.index.items():
            faces.append({
                "name": safe_name,
                "display_name": data.get("display_name", safe_name),
                "face_path": data["face_path"],
                "embedding_path": data["embedding_path"],
            })
        return faces

    def load_embedding(self, name: str) -> Optional[np.ndarray]:
        """Load embedding for a specific person (safe name). Returns 128D array or None."""
        if name in self.index:
            embedding_path = self.index[name]["embedding_path"]
            if os.path.exists(embedding_path):
                return np.load(embedding_path)
        return None

    def load_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Load all saved embeddings into memory. Returns Dict mapping safe_name to embedding."""
        embeddings = {}
        for safe_name in self.index.keys():
            embedding = self.load_embedding(safe_name)
            if embedding is not None:
                embeddings[safe_name] = embedding
        return embeddings

    def find_closest_match(
        self, query_embedding: np.ndarray, threshold: float = 0.6
    ) -> Optional[Tuple[str, float]]:
        """
        Find the closest matching face in the database.
        Returns (display_name, distance) if match found, else None.
        """
        embeddings = self.load_all_embeddings()
        if not embeddings:
            return None
        best_match = None
        best_distance = float("inf")
        for name, saved_embedding in embeddings.items():
            distance = np.linalg.norm(query_embedding - saved_embedding)
            if distance < best_distance:
                best_distance = distance
                best_match = name
        if best_distance <= threshold:
            display_name = self.index[best_match].get("display_name", best_match)
            return (display_name, float(best_distance))
        return None

    def rename_face(self, old_name: str, new_name: str) -> bool:
        """Rename a saved face (display name only). old_name is safe name."""
        if old_name not in self.index:
            return False
        self.index[old_name]["display_name"] = new_name
        self._save_index()
        return True

    def delete_face(self, name: str) -> bool:
        """Delete a saved face and its data. name is safe name."""
        if name not in self.index:
            return False
        person_dir = self.faces_dir / name
        if person_dir.exists():
            shutil.rmtree(person_dir)
        del self.index[name]
        self._save_index()
        return True

    def search_faces(self, query: str) -> List[Dict]:
        """Search for faces by name (case-insensitive substring match)."""
        query_lower = query.lower()
        results = []
        for face in self.get_all_faces():
            if query_lower in face["display_name"].lower():
                results.append(face)
        return results
