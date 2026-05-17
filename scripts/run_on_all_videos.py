import glob
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

#for video in sorted(glob.glob("video/output*.mp4")):
#    print(f"Processing: {video}")
#    model.predict(source=video, save=True, vid_stride=5)

#for video in sorted(glob.glob("video/tennis_video.mp4")):
#    print(f"Processing: {video}")
#    model.predict(source=video, save=True, vid_stride=5)

#for video in sorted(glob.glob("video/semifinal2015.mp4")):
#    print(f"Processing: {video}")
#    model.predict(source=video, save=True, vid_stride=5)

#videos_to_run = [
#    "video/ATPMadrid_1_2026_fixed.mp4",
#    "video/ATPMadrid_2_2026_fixed.mp4",
#    "video/ATPMadrid_3_2026_fixed.mp4"
#]

#for video in videos_to_run:
#    print(f"Processing: {video}")
#    model.predict(source=video, save=True, vid_stride=5)


# Run ONLY all parts of ATPMadrid_1_2026
#for video in sorted(glob.glob("video/ATP/ATPMadrid_2_2026_part_*.mp4")):
#    print(f"Processing: {video}")
#    model.predict(source=video, save=True, vid_stride=5)

#for video in sorted(glob.glob("video/ATP/ATPMadrid_3_2026_part_*.mp4")):
#    print(f"Processing: {video}")
#    model.predict(source=video, save=True, vid_stride=5)


videos_to_run = [
    "video/ATP/ATPMadrid_1_2026_part_*"
]

for video in videos_to_run:
    print(f"Processing: {video}")

    model.predict(
        source=video,
        save=True,
        save_crop=True,
        save_txt=True,
        vid_stride=5,
        project="runs/detect",
        name="ocr_ready",
        exist_ok=True
    )