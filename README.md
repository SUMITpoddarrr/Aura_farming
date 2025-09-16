# Aura Farming ✨

*A real-time “aura” visualizer that detects emotions from your face and renders a glowing halo around you — using OpenCV, MediaPipe, and DeepFace / FER.*

---

## Features
- Detects emotions in real time using:
  - [FER](https://github.com/justinshenk/fer) (fast, lightweight)
  - [DeepFace](https://github.com/serengil/deepface) (more accurate, slower)
- Dynamic glowing aura around the user’s silhouette.
- Smooth color transitions based on detected emotion.
- Adjustable aura size, intensity & pulse animation.
- Option to switch models on the fly (`m` key).
- Save a screenshot of the current frame (`s` key).
- Modular code structure for easier debugging and updates.

---

## Installation

1. **Clone the repository**

   bash
   git clone https://github.com/SUMITpoddarrr/Aura_farming.git
   cd Aura_farming
2. **Create and activate a virtual environment**
   python -m venv venv
   venv\Scripts\activate
3. **Install dependencies**
   pip install -r requirements.txt
4. **Install dependencies**
   python -m aura.main.py

## Demo
[▶️ Watch the demo](result.mp4)




