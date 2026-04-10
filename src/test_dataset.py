from dataset import OpenImagesSegmentationDataset


def main():
    dataset = OpenImagesSegmentationDataset(
        index_path="data/processed/train_index.json",
        image_size=(256, 256),
    )

    print("Dataset size:", len(dataset))

    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Mask unique values:", mask.unique().tolist())


if __name__ == "__main__":
    main()
