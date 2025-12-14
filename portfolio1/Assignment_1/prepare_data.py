from data_utils.data_formatter import run_data_parser
from data_utils.data_splitter import run_data_splitter

def main():
    print("--- Starting Data Preparation Pipeline ---")

    print("1. Running data formatter...")
    try:
        run_data_parser()
        print("   -> Annotation file created.")
    except Exception as e:
        print(f"   -> ERROR during formatting: {e}")
        return

    print("2. Running data splitter...")
    try:
        run_data_splitter()
        print("   -> Train, Val, Test ID files created.")
    except Exception as e:
        print(f"   -> ERROR during splitting: {e}")
        return

    print("--- Data Preparation Complete ---")

if __name__ == "__main__":
    main()