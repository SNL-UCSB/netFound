import os
import argparse
import pyarrow as pa
import pyarrow.ipc as ipc


def merge_arrow_files(input_folder, output_file):
    # Get all the Arrow files in the specified folder
    input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.arrow')]
    print(f"Found {len(input_files)} Arrow files in the folder.")

    # get schema
    first_file = input_files[0]
    with pa.memory_map(first_file, 'r') as source:
        reader = ipc.open_stream(source)
        schema = reader.schema

    # Initialize the output stream
    with pa.OSFile(output_file, 'wb') as sink:
        with ipc.new_stream(sink, schema) as writer:
            for input_file in input_files:
                with pa.memory_map(input_file, 'r') as source:
                    reader = ipc.open_stream(source)
                    for batch in reader:
                        writer.write_batch(batch)


def main():
    parser = argparse.ArgumentParser(description="Merge Apache Arrow streaming files.")
    parser.add_argument("input_folder", type=str, help="The folder containing the Arrow streaming files.")
    parser.add_argument("output_file", type=str, help="The output Arrow streaming file.")

    args = parser.parse_args()

    merge_arrow_files(args.input_folder, args.output_file)


if __name__ == "__main__":
    main()
