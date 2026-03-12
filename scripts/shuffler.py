import sys
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

def shuffle_arrow_file(input_path, output_path, seed=42):
    with pa.memory_map(input_path, 'r') as source:
        reader = ipc.open_stream(source)
        table = reader.read_all()

    num_rows = table.num_rows
    print(f"Loaded table with {num_rows} rows.")
    np.random.seed(seed)
    perm = np.random.permutation(num_rows)
    shuffled_table = table.take(pa.array(perm))
    
    print(f"Shuffled table with seed {seed}.")
    
    with pa.OSFile(output_path, 'wb') as sink:
        writer = ipc.new_stream(sink, shuffled_table.schema)
        writer.write_table(shuffled_table)
        writer.close()
    
    print(f"Shuffled file written to {output_path}.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python shuffler.py <input_file> <output_file> [seed]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    
    shuffle_arrow_file(input_file, output_file, seed)
