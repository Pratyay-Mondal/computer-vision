import pickle
import os

OUTPUT_FILE = 'formatted_annotations.pkl' # Ensure this matches the file name

try:
    # 1. Open the file in binary read mode ('rb')
    with open(OUTPUT_FILE, 'rb') as f:
        # 2. Load (unpickle) the object from the file
        all_records = pickle.load(f)

    # 3. Print a summary
    print(f"Total number of image records found: {len(all_records)}\n")

    if all_records:
        import json
        print(json.dumps(all_records[0], indent=4))
    else:
        print("The loaded list is empty.")

except FileNotFoundError:
    print(f"❌ Error: The file '{OUTPUT_FILE}' was not found.")
except Exception as e:
    print(f"❌ An error occurred during unpickling: {e}")