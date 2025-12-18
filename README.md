# Car Detection on BDD100K using YOLOv8

This project implements a **single-class car detection pipeline** on the BDD100K dataset using **Ultralytics YOLOv8**.  
The workflow includes annotation conversion, dataset analysis, downsampling, model fine-tuning, and evaluation on validation and test splits.

---

## Dataset
- **Dataset**: BDD100K
- **Task**: Car detection (single class)
- **Original annotations**: JSON format
- **Converted to**: YOLO format (`.txt`)

> Due to size constraints, the BDD100K dataset (images and labels) is **not included** in this repository.

Expected dataset structure after download and preprocessing:


bdd100k_images_100k/100k/{train,val,test}/
bdd100k_labels/100k/{train,val,test}/


## Preprocessing
The following preprocessing steps are performed:

1. **JSON → YOLO label conversion**  
   Extracts only `car` bounding boxes and converts them into YOLO format.
2. **Dataset analysis**
   - Object size distribution
   - Truncated vs non-truncated objects
   - Time-of-day vs weather distribution
3. **Downsampling**
   Training data is downsampled using stratified sampling (by time-of-day) to reduce compute cost while preserving data diversity.

Scripts are available under:
