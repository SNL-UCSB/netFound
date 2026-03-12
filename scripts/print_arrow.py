import pyarrow as pa
import pyarrow.ipc as ipc
import sys

def read_and_print_labels(file_path):
    try:
        # Open the Arrow file as a stream
        with pa.OSFile(file_path, 'rb') as source:
            reader = ipc.open_stream(source)
            
            # Read all batches in the stream
            for batch in reader:
                # Convert the batch to a table
                table = pa.Table.from_batches([batch])
                print(table)
                
                # Get the "label" field if it exists
                if 'labels' in table.column_names:
                    label_column = table['labels']
                    
                    # Print the first 10 rows of the "label" column
                    print("First 10 'label' values:")
                    print(label_column.to_pylist()[:10])
                else:
                    print("'labels' field not found in the file.")
                break  # Exit after processing the first batch
    except Exception as e:
        print(f"Error processing the file: {e}")

if __name__ == "__main__":
    # Ensure a filename is provided as the first argument
    if len(sys.argv) < 2:
        print("Usage: python print_arrow.py <arrow_file>")
        sys.exit(1)
    
    # Get the filename from the command-line argument
    file_path = sys.argv[1]

    # Call the function with the provided filename
    read_and_print_labels(file_path)
