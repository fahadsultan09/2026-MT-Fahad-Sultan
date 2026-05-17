import os

def check_dataset(path):
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(path, split, "images")
        lbl_dir = os.path.join(path, split, "labels")

        print(f"\nChecking {split} split:")
        print(f"Images: {len(os.listdir(img_dir))}")
        print(f"Labels: {len(os.listdir(lbl_dir))}")

if __name__ == "__main__":
    check_dataset("datasets/scoreboardtennis_v1")
