import argparse
import math
import os

from datasets import Dataset

from veomni.data.multimodal.image_utils import load_image_bytes_from_path


def load_dataset(dataset_path: str):
    captions_file = os.path.join(dataset_path, "captions.txt")
    images_file = os.path.join(dataset_path, "images.txt")

    with open(captions_file, encoding="utf-8") as f:
        captions = f.readlines()

    with open(images_file, encoding="utf-8") as f:
        image_paths = f.readlines()

    captions = [caption.strip() for caption in captions]
    image_paths = [image_path.strip() for image_path in image_paths]

    assert len(captions) == len(image_paths), (
        f"captions.txt {len(captions)} and images.txt {len(image_paths)} lines do not match"
    )

    data = {"text": captions, "image": image_paths}
    dataset = Dataset.from_dict(data)
    return dataset


def convert_to_parquet(dataset_path: str, output_dir: str):
    NUM_SHARD = 30
    NUM_PROC = 32

    dataset = load_dataset(dataset_path)

    os.makedirs(output_dir, exist_ok=True)
    total_len = len(dataset)
    batch_len = math.ceil(total_len / NUM_SHARD)
    print(f"Total length: {total_len}, batch length: {batch_len}")

    index = 0
    for i in range(0, total_len, batch_len):
        print(f"Generating {index}th parquet file")
        end_idx = min(i + batch_len, total_len)
        chunk_ds = dataset.select(range(i, end_idx))
        chunk_num_proc = min(NUM_PROC, len(chunk_ds))

        def process_example(example):
            image_bytes = load_image_bytes_from_path(os.path.join(dataset_path, example["image"]))
            return {
                "text": example["text"],
                "image_bytes": image_bytes,
                "source": "X2I-text-to-image",
            }

        ds = chunk_ds.map(
            process_example,
            num_proc=chunk_num_proc,
            remove_columns=chunk_ds.column_names,
            keep_in_memory=True,
            desc=f"Processing shard {index}",
        )
        ds.to_parquet(os.path.join(output_dir, f"{index}.parquet"))
        index += 1


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True)
    args.add_argument("--output_dir", type=str, required=True)
    args = args.parse_args()
    convert_to_parquet(args.dataset_path, args.output_dir)
