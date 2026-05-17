🎾 Tennis Scoreboard Detection & Game State Extraction
A Machine Learning Thesis Project by Fahad Sultan

📌 Overview
This thesis project focuses on automatically extracting tennis scoreboard information from broadcast match videos using YOLOv8 and a custom post‑processing pipeline.
The system detects scoreboard regions, extracts text using OCR, interprets the game state, and outputs structured data for analysis.

The repository includes:

Dataset preparation tools

YOLOv8 training configuration

Scoreboard detection + OCR pipeline

Game state reconstruction logic

Evaluation scripts

Thesis documentation support

📁 Project Structure
Code
2026-MT-Fahad-Sultan/
├── data/
│   ├── raw/                # XML annotations, raw metadata
│   ├── processed/          # Cleaned/converted data
│
├── datasets/
│   ├── scoreboardtennis_v1/  # YOLO dataset (images excluded)
│   └── data.yaml
│
├── models/                 # Model configs (no .pt files)
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── run_on_all_videos.py
│   └── evaluate.py
│
├── src/
│   ├── scoreboard/
│   │   ├── efficient_score_extractor.py
│   │   ├── process_scoreboard.py
│   ├── game/
│   │   └── process_game.py
│   └── main.py
│
├── video/                  # Input videos (ignored)
├── .gitignore
├── requirements.txt
└── README.md
🚀 Features
YOLOv8-based scoreboard detection

OCR extraction of player names and scores

Game state reconstruction (points → games → sets)

Batch video processing

XML/JSON output generation

Modular and extendable pipeline

🧠 Model Training (YOLOv8)
Training was performed using:

Dataset: scoreboardtennis_v1

Base model: yolov8s.pt

Epochs: 100+

Image size: 640

Augmentations: rotation, brightness, cropping

Training command:

bash
yolo detect train data=datasets/scoreboardtennis_v1/data.yaml model=yolov8s.pt epochs=100 imgsz=640
Trained weights are not included in this repository.
Place your weights here if needed:

Code
models/weights/best.pt
🧩 Pipeline Overview
1. Scoreboard Detection
YOLOv8 detects the scoreboard region in each frame.

2. OCR Extraction
The detected region is passed to OCR (Tesseract or EasyOCR).

3. Score Parsing
efficient_score_extractor.py converts OCR text into structured fields:

Player names

Points

Games

Sets

4. Game Logic Reconstruction
process_game.py rebuilds the match flow:

Point progression

Game winners

Set winners

5. Output
Results are saved as:

XML

JSON

Annotated video (optional)

▶️ How to Run
1. Install dependencies
bash
pip install -r requirements.txt
2. Run detection + extraction
bash
python src/main.py --video path/to/video.mp4
3. Batch process all videos
bash
python scripts/run_on_all_videos.py
4. Evaluate results
bash
python scripts/evaluate.py
📊 Results Summary
High accuracy in scoreboard localization

Robust OCR extraction after preprocessing

Correct reconstruction of game flow in most test videos

Identified failure cases: glare, motion blur, scoreboard style changes

Full results are included in the thesis document.

📚 Thesis Document
The final thesis PDF will be added to this repository once completed.

📝 License
This project is for academic use.
For commercial use, please contact the author.

🙌 Acknowledgements
Ultralytics YOLOv8

Open-source OCR libraries

Supervisors and academic support
