import json
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# -----------------------------
# CONFIG (edit if you want)
# -----------------------------
SEED = 42
TRAIN_FRACTION = 0.20     # 20% train sampling (change as needed)
KEEP_FULL_VAL = True      # keep full val (recommended)
COPY_MODE = "copy"        # "copy" or "symlink" (symlink is faster, copy is safer)

# Your current folders (from your screenshot)
IMAGES_ROOT = Path("bdd100k_images_100k/100k")   # train/ val/ test/
LABELS_ROOT = Path("bdd100k_labels/100k")        # train/ val/ test/ (has .json and .txt)
SPLITS = ["train", "val", "test"]

# Output smaller dataset (YOLO-friendly)
OUT_ROOT = Path("bdd_small")   # will create bdd_small/images/{split}, bdd_small/labels/{split}


# -----------------------------
# Helpers
# -----------------------------
def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def get_objects(item: dict):
    # Your JSON sample structure: frames[0].objects
    if "frames" in item and item["frames"]:
        return item["frames"][0].get("objects", [])
    # Some variants might have "labels"
    if "labels" in item and isinstance(item["labels"], list):
        return item["labels"]
    return []

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def car_box_norm_area(box2d, w, h):
    x1, y1, x2, y2 = box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
    x1 = clamp(x1, 0, w - 1)
    x2 = clamp(x2, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    y2 = clamp(y2, 0, h - 1)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 0 or bh <= 0:
        return None
    return (bw * bh) / (w * h)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_or_symlink(src: Path, dst: Path, mode="copy"):
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)

def plot_hist(data, title, xlabel, bins=50):
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_stacked_bar(ct: pd.DataFrame, title):
    ax = ct.plot(kind="bar", stacked=True)
    ax.set_title(title)
    ax.set_xlabel("timeofday")
    ax.set_ylabel("image count")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 1) ANALYZE: size hist + crosstab
# -----------------------------
def analyze_split(split: str):
    labels_dir = LABELS_ROOT / split
    images_dir = IMAGES_ROOT / split

    json_files = list(labels_dir.glob("*.json"))
    print(f"[{split}] scanning {len(json_files)} json files...")

    # counts
    weather = Counter()
    timeofday = Counter()
    scene = Counter()

    # combo: (weather, timeofday) for crosstab
    time_weather = Counter()

    # car sizes
    car_norm_areas = []

    # keep per-image metadata (for downsampling)
    rows = []

    for jf in json_files:
        data = read_json(jf)
        if data is None:
            continue

        name = data.get("name")
        if not name:
            continue

        attrs = data.get("attributes", {})
        wthr = attrs.get("weather", "UNKNOWN")
        tod = attrs.get("timeofday", "UNKNOWN")
        scn = attrs.get("scene", "UNKNOWN")

        weather[wthr] += 1
        timeofday[tod] += 1
        scene[scn] += 1
        time_weather[(tod, wthr)] += 1

        # find image (needed for width/height to compute normalized area)
        img_path = images_dir / f"{name}.jpg"
        if not img_path.exists():
            img_path = images_dir / f"{name}.png"
        if not img_path.exists():
            # skip size computation if image missing
            rows.append({"name": name, "split": split, "timeofday": tod, "weather": wthr, "scene": scn,
                         "has_image": False})
            continue

        try:
            iw, ih = Image.open(img_path).size
        except Exception:
            rows.append({"name": name, "split": split, "timeofday": tod, "weather": wthr, "scene": scn,
                         "has_image": False})
            continue

        # car sizes: iterate objects, keep only category == car and has box2d
        objs = get_objects(data)
        for obj in objs:
            if obj.get("category") != "car":
                continue
            box2d = obj.get("box2d")
            if not box2d:
                continue
            na = car_box_norm_area(box2d, iw, ih)
            if na is not None:
                car_norm_areas.append(na)

        rows.append({"name": name, "split": split, "timeofday": tod, "weather": wthr, "scene": scn,
                     "has_image": True})

    # tables
    weather_df = pd.DataFrame(weather.most_common(), columns=["weather", "count"])
    tod_df = pd.DataFrame(timeofday.most_common(), columns=["timeofday", "count"])
    scene_df = pd.DataFrame(scene.most_common(), columns=["scene", "count"])

    # crosstab
    tw_rows = [{"timeofday": tod, "weather": wthr, "count": c} for (tod, wthr), c in time_weather.items()]
    tw_df = pd.DataFrame(tw_rows)
    ct = tw_df.pivot_table(index="timeofday", columns="weather", values="count",
                           aggfunc="sum", fill_value=0)

    meta_df = pd.DataFrame(rows)

    return {
        "weather_df": weather_df,
        "tod_df": tod_df,
        "scene_df": scene_df,
        "ct": ct,
        "car_norm_areas": car_norm_areas,
        "meta_df": meta_df
    }


# -----------------------------
# 2) DOWNSAMPLE TRAIN (stratified by timeofday)
# -----------------------------
def downsample_train(train_meta: pd.DataFrame, frac: float):
    # keep only those with images present
    df = train_meta[train_meta["has_image"] == True].copy()

    random.seed(SEED)

    # stratified by timeofday: sample same fraction inside each group
    parts = []
    for tod, g in df.groupby("timeofday"):
        n = max(1, int(len(g) * frac))
        parts.append(g.sample(n=n, random_state=SEED))

    sampled = pd.concat(parts, ignore_index=True)
    return sampled


# -----------------------------
# 3) BUILD SMALL DATASET FOLDERS (images + labels .txt)
# -----------------------------
def build_small_split(split: str, names, copy_mode="copy"):
    out_images = OUT_ROOT / "images" / split
    out_labels = OUT_ROOT / "labels" / split
    ensure_dir(out_images)
    ensure_dir(out_labels)

    src_images = IMAGES_ROOT / split
    src_labels = LABELS_ROOT / split

    # We will copy/symlink:
    # - image file: name.jpg or name.png
    # - yolo label file: name.txt   (your converter already created these)
    missing_img = 0
    missing_txt = 0

    for name in names:
        # image
        img = src_images / f"{name}.jpg"
        if not img.exists():
            img = src_images / f"{name}.png"
        if img.exists():
            copy_or_symlink(img, out_images / img.name, mode=copy_mode)
        else:
            missing_img += 1

        # label .txt
        txt = src_labels / f"{name}.txt"
        if txt.exists():
            copy_or_symlink(txt, out_labels / txt.name, mode=copy_mode)
        else:
            missing_txt += 1

    print(f"[small {split}] wrote {len(names)} samples | missing_img={missing_img} missing_txt={missing_txt}")


def main():
    random.seed(SEED)

    # --- Analyze each split ---
    results = {}
    for split in SPLITS:
        results[split] = analyze_split(split)

        print(f"\n=== {split.upper()} ===")
        print("Top weather:\n", results[split]["weather_df"].head(10))
        print("Top timeofday:\n", results[split]["tod_df"].head(10))
        print("Top scene:\n", results[split]["scene_df"].head(10))
        print("\nTime of day vs Weather crosstab:\n", results[split]["ct"])

    # --- Plots: object size histogram (cars) per split ---
    for split in SPLITS:
        areas = results[split]["car_norm_areas"]
        if len(areas) == 0:
            print(f"[{split}] No car boxes found for size histogram.")
            continue
        plot_hist(
            areas,
            title=f"Car object size distribution (normalized area) — {split}",
            xlabel="(bbox area) / (image area)",
            bins=50
        )

    # --- Plots: timeofday vs weather crosstab (stacked bar) per split ---
    for split in SPLITS:
        ct = results[split]["ct"]
        if ct.empty:
            continue
        plot_stacked_bar(ct, title=f"Time of day vs Weather (stacked counts) — {split}")

    # --- Downsample train, keep full val, ignore test for training (but you can make a small test too) ---
    train_meta = results["train"]["meta_df"]
    sampled_train = downsample_train(train_meta, TRAIN_FRACTION)
    train_names = sampled_train["name"].tolist()

    if KEEP_FULL_VAL:
        val_names = results["val"]["meta_df"][results["val"]["meta_df"]["has_image"] == True]["name"].tolist()
    else:
        # optionally downsample val too (e.g., same fraction)
        val_meta = results["val"]["meta_df"]
        val_names = downsample_train(val_meta, TRAIN_FRACTION)["name"].tolist()

    # For test: usually not needed for training; but we can create a small test set
    test_meta = results["test"]["meta_df"]
    test_names = test_meta[test_meta["has_image"] == True]["name"].tolist()
    # optionally shrink test to 20% so it’s small
    test_names = random.sample(test_names, k=max(1, int(len(test_names) * 0.20))) if len(test_names) else []

    print("\n--- Building smaller dataset folders ---")
    build_small_split("train", train_names, copy_mode=COPY_MODE)
    build_small_split("val", val_names, copy_mode=COPY_MODE)
    build_small_split("test", test_names, copy_mode=COPY_MODE)

    # Write a YOLO data.yaml for the small dataset
    yaml_path = OUT_ROOT / "bdd_car_small.yaml"
    ensure_dir(yaml_path.parent)
    yaml_path.write_text(
        "path: .\n"
        "train: bdd_small/images/train\n"
        "val: bdd_small/images/val\n"
        "test: bdd_small/images/test\n\n"
        "names:\n"
        "  0: car\n"
    )
    print(f"\nWrote YOLO yaml: {yaml_path}")
    print("\nDone ✅")

if __name__ == "__main__":
    main()
