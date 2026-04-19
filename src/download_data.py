import fiftyone as fo
import fiftyone.zoo as foz


def download_dataset():
    fo.config.dataset_zoo_dir = "./data/openimages"

    classes = ["Car", "Bus", "Truck"]

    print("Downloading training dataset...")
    train_dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["segmentations"],
        classes=classes,
        max_samples=3000,
    )

    print("Downloading test dataset...")
    test_dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["segmentations"],
        classes=classes,
        max_samples=1000,
    )

    print("Download complete!")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = download_dataset()

    session = fo.launch_app(train_dataset)
    session.wait()
