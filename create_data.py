"""Create a data stream for training.

Reads a CSV file whose first few lines might be something like:

object,shape,color,material,size  
coin,circular,silver,metal,small  
book,rectangular,blue,paper,medium

Generates a stream of sentences like:

The coin is circular.  
The book is paper.  
The coin is metal.

The intention is that the stream of sentences will be used as a training stream
for a transformer model, after suitable tokenization.
"""

import argparse
import csv
import os

from seed_utils import make_rng

# Constants
CSV_DIR = os.path.join(os.path.dirname(__file__), 'item_metadata')
CSV_FILENAME = 'items-10.csv'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_FILENAME = 'stream.txt'
SEED = 1234
STREAM_BYTE_TARGET = 1024  # 1 KiB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a data stream for training.")
    parser.add_argument(
        "-d", "--csv-dir",
        default=CSV_DIR,
        help=f"Directory containing the input CSV file. Default: {CSV_DIR}",
    )
    parser.add_argument(
        "-f", "--csv-filename",
        default=CSV_FILENAME,
        help=f"Name of the input CSV file. Default: {CSV_FILENAME}",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=OUTPUT_DIR,
        help=f"Directory to write the output stream file. Default: {OUTPUT_DIR}",
    )
    parser.add_argument(
        "-O", "--output-filename",
        default=OUTPUT_FILENAME,
        help=f"Name of the output stream file. Default: {OUTPUT_FILENAME}",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=SEED,
        help=f"Random number seed. Default: {SEED}",
    )
    parser.add_argument(
        "-n", "--stream-byte-target",
        type=int,
        default=STREAM_BYTE_TARGET,
        help=f"Approximate target size of the output stream in bytes. Default: {STREAM_BYTE_TARGET}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Read the CSV file
    csv_path = os.path.join(args.csv_dir, args.csv_filename)
    items = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)

    # Attributes to randomly select from (excluding 'object')
    attributes = ['shape', 'color', 'material', 'size']

    # Generate stream of sentences
    stream = []
    stream_length = 0
    rng = make_rng(args.seed)

    while stream_length < args.stream_byte_target:
        # Pick a random item and attribute
        item = rng.choice(items)
        attribute = rng.choice(attributes)

        # Create sentence: "The {object} is {attribute_value}."
        sentence = f"The {item['object']} is {item[attribute]}."

        stream.append(sentence)
        stream_length += len(sentence) + 1  # +1 for newline

    # Write to file
    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, args.output_filename)
    with open(output_path, 'w', newline='\n', encoding='utf-8') as f:
        for sentence in stream:
            f.write(sentence + '\n')

    print(f"Generated stream with {len(stream)} sentences and {stream_length} bytes")
    print(f"Seed: {args.seed}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
