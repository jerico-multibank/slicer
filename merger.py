import pandas as pd
import argparse
import os
import sys
from slicer import load_dataframe, save_dataframe  # reaproveita funções do slicer

def join_dataframes(left_file, right_file, on_columns, how, output_file, suffixes=None):
    """
    Join two dataframes based on specified columns.
    
    Args:
        left_file: Path to the left (primary) file
        right_file: Path to the right (secondary) file
        on_columns: Column(s) to join on
        how: Type of join ('left', 'right', 'inner', 'outer')
        output_file: Path to save the merged result
        suffixes: Tuple of suffixes for overlapping columns
    """
    try:
        print(f"Loading {left_file}...")
        df1 = load_dataframe(left_file)
        print(f"Left file loaded: {df1.shape[0]} rows, {df1.shape[1]} columns")
        
        print(f"Loading {right_file}...")
        df2 = load_dataframe(right_file)
        print(f"Right file loaded: {df2.shape[0]} rows, {df2.shape[1]} columns")

        # Parse column names
        if ',' in on_columns:
            on_columns = [col.strip() for col in on_columns.split(',')]
        else:
            on_columns = on_columns.strip()

        # Verify columns exist
        missing_cols_left = []
        missing_cols_right = []
        
        if isinstance(on_columns, list):
            for col in on_columns:
                if col not in df1.columns:
                    missing_cols_left.append(col)
                if col not in df2.columns:
                    missing_cols_right.append(col)
        else:
            if on_columns not in df1.columns:
                missing_cols_left.append(on_columns)
            if on_columns not in df2.columns:
                missing_cols_right.append(on_columns)
        
        if missing_cols_left or missing_cols_right:
            print("ERROR: Missing columns!")
            if missing_cols_left:
                print(f"Left file missing: {missing_cols_left}")
                print(f"Available columns in left file: {list(df1.columns)}")
            if missing_cols_right:
                print(f"Right file missing: {missing_cols_right}")
                print(f"Available columns in right file: {list(df2.columns)}")
            return False

        # Set default suffixes if not provided
        if suffixes is None:
            suffixes = ('_left', '_right')

        # Perform the merge
        print(f"Performing {how} join on column(s): {on_columns}")
        merged_df = pd.merge(df1, df2, on=on_columns, how=how, suffixes=suffixes)
        
        # Show join statistics
        print(f"Join completed:")
        print(f"  - Original left: {df1.shape[0]} rows")
        print(f"  - Original right: {df2.shape[0]} rows")
        print(f"  - Merged result: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        
        # Check for potential issues
        if how == 'left' and merged_df.shape[0] != df1.shape[0]:
            print(f"WARNING: Left join resulted in different row count. Expected {df1.shape[0]}, got {merged_df.shape[0]}")
        
        # Save result
        save_dataframe(merged_df, output_file)
        print(f"Merged result saved to '{output_file}'")
        
        return True
        
    except Exception as e:
        print(f"ERROR during join operation: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Join two Excel/CSV files based on key column(s) (like VLOOKUP).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --left_file data1.xlsx --right_file data2.xlsx --on ID
  %(prog)s --left_file data1.csv --right_file data2.csv --on "ID1,ID2" --how inner
  %(prog)s --left_file base.xlsx --right_file lookup.xlsx --on CustomerID --output_file result.xlsx
        """
    )
    
    parser.add_argument("--left_file", required=True, 
                       help="Path to the left base file (primary)")
    parser.add_argument("--right_file", required=True, 
                       help="Path to the right base file (secondary)")
    parser.add_argument("--on", required=True, 
                       help="Column(s) to join on. Examples: 'ID' or 'ID1,ID2'")
    parser.add_argument("--how", default="left", 
                       choices=["left", "right", "inner", "outer"],
                       help="Type of join to perform (default: left)")
    parser.add_argument("--output_file", default="joined_output.xlsx", 
                       help="Path to save the merged result (default: joined_output.xlsx)")
    parser.add_argument("--suffixes", default="_left,_right",
                       help="Suffixes for overlapping columns (default: _left,_right)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.left_file):
        print(f"ERROR: Left file '{args.left_file}' not found.")
        sys.exit(1)
    
    if not os.path.exists(args.right_file):
        print(f"ERROR: Right file '{args.right_file}' not found.")
        sys.exit(1)

    # Parse suffixes
    suffixes = tuple(args.suffixes.split(',')) if ',' in args.suffixes else ('_left', '_right')

    # Perform the join
    success = join_dataframes(
        args.left_file, 
        args.right_file, 
        args.on, 
        args.how, 
        args.output_file,
        suffixes
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()