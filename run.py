#!/usr/bin/env python3
"""
Launcher script for Face Recognition App.
Run this from the project root directory.
"""
import sys
import os

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_path)

# Import main function from main module (main.py in src/)
from main import main

if __name__ == "__main__":
    print("Starting Face Recognition App...")
    print("Press Ctrl+C to exit")
    main()
