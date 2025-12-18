import json
from pathlib import Path
from PIL import Image

def yolo_from_box2d(b, w, h):
    x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]

    x1 = max(0.0, min(x1, w - 1))
    x2 = max(0.0, min(x2, w - 1))
    y1 = max(0.0, min(y1, h - 1))
    y2 = max(0.0, min(y2, h - 1))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 0 or bh <= 0:
        return None

    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h

def extract_objects(item):
    # your format: frames -> objects
    if "frames" in item and item["frames"]:
        return item["frames"][0].get("objects", [])
    return []

def convert_split(split):
    images_dir = Path(f"bdd100k_images_100k/100k/{split}")
    labels_dir = Path(f"bdd100k_labels/100k/{split}")
    labels_dir.mkdir(parents=True, exist_ok=True)

    # if each image has its own json file
    json_files = list(labels_dir.glob("*.json"))

    print(f"[{split}] Found {len(json_files)} json files")

    for json_file in json_files:
        data = json.loads(json_file.read_text())
        name = data.get("name")
        if not name:
            continue

        # image path
        img_path = images_dir / f"{name}.jpg"
        if not img_path.exists():
            img_path = images_dir / f"{name}.png"
        if not img_path.exists():
            print(f" Missing image for {name}")
            continue

        w, h = Image.open(img_path).size

        yolo_lines = []
        for obj in extract_objects(data):
            if obj.get("category") != "car":
                continue
            box2d = obj.get("box2d")
            if not box2d:
                continue

            y = yolo_from_box2d(box2d, w, h)
            if y is None:
                continue

            xc, yc, bw, bh = y
            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # overwrite json → yolo txt
        (labels_dir / f"{name}.txt").write_text("\n".join(yolo_lines))

    print(f"[{split}] Done.")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        convert_split(split)
