from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data="datasets/scoreboardtennis_v1/data.yaml",
        epochs=50,
        imgsz=640,
        project="models",
        name="exp1"
    )

if __name__ == "__main__":
    main()
