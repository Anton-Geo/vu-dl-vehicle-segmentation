from pathlib import Path
import pandas as pd
import json
import random


CLASS_MAP = {
    "/m/0k4j": 1,    # Car
    "/m/01bjv": 2,   # Bus
    "/m/07r04": 3,   # Truck
}

CLASS_NAMES = {
    0: "background",
    1: "car",
    2: "bus",
    3: "truck",
}


def build_index(split: str):
    base = Path("data/openimages/open-images-v7") / split
    images_dir = base / "data"
    masks_dir = base / "labels" / "masks"
    seg_csv = base / "labels" / "segmentations.csv"

    print(f"\nProcessing {split}...")

    df = pd.read_csv(seg_csv)
    df = df[df["LabelName"].isin(CLASS_MAP.keys())].copy()

    image_files = {p.stem: p for p in images_dir.glob("*.jpg")}
    df = df[df["ImageID"].isin(image_files.keys())].copy()

    print(f"Filtered rows: {len(df)}")

    mask_files = {p.name: p for p in masks_dir.rglob("*.png")}
    print(f"Indexed {len(mask_files)} masks")

    records = []

    for image_id, group in df.groupby("ImageID"):
        image_path = image_files.get(image_id)
        if image_path is None:
            continue

        objects = []
        for _, row in group.iterrows():
            mask_path = mask_files.get(row["MaskPath"])
            if mask_path is None:
                continue

            class_id = CLASS_MAP[row["LabelName"]]

            objects.append(
                {
                    "mask_path": str(mask_path),
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                }
            )

        if not objects:
            continue

        records.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "objects": objects,
            }
        )

    print(f"Collected {len(records)} image records")
    return records


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def split_train_val(records, val_ratio=0.2, seed=42):
    random.seed(seed)
    random.shuffle(records)

    val_size = int(len(records) * val_ratio)

    val = records[:val_size]
    train = records[val_size:]

    return train, val


if __name__ == "__main__":
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = build_index("train")
    test_records = build_index("validation")

    train_split, val_split = split_train_val(train_records)

    print(f"\nFinal splits:")
    print(f"Train: {len(train_split)}")
    print(f"Val:   {len(val_split)}")
    print(f"Test:  {len(test_records)}")

    save_json(train_split, output_dir / "train_index.json")
    save_json(val_split, output_dir / "val_index.json")
    save_json(test_records, output_dir / "test_index.json")
