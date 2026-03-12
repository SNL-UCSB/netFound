# shuffle combined dataset

import argparse
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

def shuffle_dataset(input_file, output_file):
    # Read the input file
    with open(input_file, "rb") as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()

    # Shuffle the table
    num_rows = table.num_rows
    shuffled_indices = np.random.permutation(num_rows)
    shuffled_table = table.take(pa.array(shuffled_indices))

    # Write the shuffled table to the output file
    with open(output_file, "wb") as f:
        writer = ipc.new_stream(f, shuffled_table.schema)
        writer.write_table(shuffled_table)
        writer.close()


def main():
    parser = argparse.ArgumentParser(description="Shuffle Apache Arrow streaming files.")
    parser.add_argument("input_file", type=str, help="The input Arrow file with data.")
    parser.add_argument("output_file", type=str, help="The output Arrow streaming file.")

    args = parser.parse_args()

    shuffle_dataset(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
