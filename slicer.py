import pandas as pd
import argparse
import sys
import os
import glob
from typing import Dict, Any, Optional, Union, List
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataFrameFilter:
    """A class to handle DataFrame filtering operations with various operators."""
    
    VALID_OPERATORS = {
        'string': ['==', '!=', 'contains', 'startswith', 'endswith', 'notcontains', 'icontains', 'isnull', 'isnotnull'],
        'numeric': ['==', '!=', '<', '<=', '>', '>=', 'isnull', 'isnotnull'],
        'common': ['==', '!=', 'isnull', 'isnotnull']
    }
    
    def __init__(self, dataframe: pd.DataFrame):
        """Initialize with a DataFrame."""
        self.dataframe = dataframe.copy()
    
    def _validate_column(self, column_name: str) -> None:
        """Validate that column exists in DataFrame."""
        if column_name not in self.dataframe.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame. "
                           f"Available columns: {list(self.dataframe.columns)}")
    
    def _convert_value(self, value: str, column_name: str) -> Union[str, int, float]:
        """Convert string value to appropriate type based on column data type."""
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Try to convert to numeric if the column is numeric
        if pd.api.types.is_numeric_dtype(self.dataframe[column_name]):
            try:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                pass
        
        return value
    
    def _is_string_column(self, column_name: str) -> bool:
        """Check if column contains string data."""
        return pd.api.types.is_string_dtype(self.dataframe[column_name]) or \
               pd.api.types.is_object_dtype(self.dataframe[column_name])
    
    def _is_numeric_column(self, column_name: str) -> bool:
        """Check if column contains numeric data."""
        return pd.api.types.is_numeric_dtype(self.dataframe[column_name])
    
    def single_filter(self, column_name: str, operator: str, value: Any) -> pd.DataFrame:
        """
        Apply a single filter condition to the DataFrame.
        
        Args:
            column_name: Name of the column to filter
            operator: Comparison operator
            value: Value to compare against (ignored for isnull/isnotnull)
            
        Returns:
            Filtered DataFrame
        """
        self._validate_column(column_name)
        
        # For null checks, we don't need to convert the value
        if operator in ['isnull', 'isnotnull']:
            try:
                if operator == 'isnull':
                    return self.dataframe[self.dataframe[column_name].isnull()]
                elif operator == 'isnotnull':
                    return self.dataframe[self.dataframe[column_name].notnull()]
            except Exception as e:
                raise ValueError(f"Error applying null filter: {str(e)}")
        
        # Convert value to appropriate type for non-null operators
        if isinstance(value, str):
            value = self._convert_value(value, column_name)
        
        is_string = self._is_string_column(column_name)
        is_numeric = self._is_numeric_column(column_name)
        
        # Validate operator for column type
        if operator not in self.VALID_OPERATORS['common']:
            if is_string and operator not in self.VALID_OPERATORS['string']:
                raise ValueError(f"Operator '{operator}' not supported for string columns")
            elif is_numeric and operator not in self.VALID_OPERATORS['numeric']:
                raise ValueError(f"Operator '{operator}' not supported for numeric columns")
        
        try:
            if operator == "==":
                return self.dataframe[self.dataframe[column_name] == value]
            elif operator == "!=":
                return self.dataframe[self.dataframe[column_name] != value]
            elif operator == "<" and is_numeric:
                return self.dataframe[self.dataframe[column_name] < value]
            elif operator == "<=" and is_numeric:
                return self.dataframe[self.dataframe[column_name] <= value]
            elif operator == ">" and is_numeric:
                return self.dataframe[self.dataframe[column_name] > value]
            elif operator == ">=" and is_numeric:
                return self.dataframe[self.dataframe[column_name] >= value]
            elif operator == "contains" and is_string:
                return self.dataframe[self.dataframe[column_name].str.contains(str(value), na=False, case=True)]
            elif operator == "icontains" and is_string:  # Case-insensitive contains
                return self.dataframe[self.dataframe[column_name].str.contains(str(value), na=False, case=False)]
            elif operator == "startswith" and is_string:
                return self.dataframe[self.dataframe[column_name].str.startswith(str(value), na=False)]
            elif operator == "endswith" and is_string:
                return self.dataframe[self.dataframe[column_name].str.endswith(str(value), na=False)]
            elif operator == "notcontains" and is_string:
                return self.dataframe[~self.dataframe[column_name].str.contains(str(value), na=False)]
            else:
                raise ValueError(f"Unsupported operator '{operator}' for column type")
                
        except Exception as e:
            raise ValueError(f"Error applying filter: {str(e)}")
    
    def filter_dataframe(self, and_filters: Optional[Dict[str, Dict[str, Any]]] = None, 
                        or_filters: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Apply multiple filter conditions to the DataFrame.
        
        Args:
            and_filters: Dictionary of AND conditions {column: {'operator': op, 'value': val}}
            or_filters: Dictionary of OR conditions {column: {'operator': op, 'value': val}}
            
        Returns:
            Filtered DataFrame
        """
        # Start with the original dataframe
        base_df = self.dataframe.copy()
        
        # If we have both AND and OR filters, we need to handle the logic carefully
        if and_filters and or_filters:
            # First apply AND filters
            and_result = base_df.copy()
            for column, filter_config in and_filters.items():
                operator = filter_config.get('operator', '==')
                value = filter_config.get('value', None)
                temp_filter = DataFrameFilter(and_result)
                and_result = temp_filter.single_filter(column, operator, value)
            
            # Then apply OR filters to the original dataframe
            or_conditions = []
            for column, filter_config in or_filters.items():
                operator = filter_config.get('operator', '==')
                value = filter_config.get('value', None)
                temp_filter = DataFrameFilter(base_df)
                filtered_df = temp_filter.single_filter(column, operator, value)
                or_conditions.append(filtered_df.index)
            
            # Combine all OR condition indices
            combined_or_indices = set()
            for indices in or_conditions:
                combined_or_indices.update(indices)
            
            # Get rows that satisfy OR conditions
            or_result = base_df[base_df.index.isin(combined_or_indices)]
            
            # Final result: intersection of AND results and OR results
            final_indices = set(and_result.index).intersection(set(or_result.index))
            result_df = base_df[base_df.index.isin(final_indices)]
        
        # If only AND filters
        elif and_filters:
            result_df = base_df.copy()
            for column, filter_config in and_filters.items():
                operator = filter_config.get('operator', '==')
                value = filter_config.get('value', None)
                temp_filter = DataFrameFilter(result_df)
                result_df = temp_filter.single_filter(column, operator, value)
        
        # If only OR filters
        elif or_filters:
            or_conditions = []
            for column, filter_config in or_filters.items():
                operator = filter_config.get('operator', '==')
                value = filter_config.get('value', None)
                temp_filter = DataFrameFilter(base_df)
                filtered_df = temp_filter.single_filter(column, operator, value)
                or_conditions.append(filtered_df.index)
            
            # Combine all OR condition indices (union of all conditions)
            combined_or_indices = set()
            for indices in or_conditions:
                combined_or_indices.update(indices)
            
            result_df = base_df[base_df.index.isin(combined_or_indices)]
        
        # If no filters
        else:
            result_df = base_df
        
        return result_df

class MultiDataFrameProcessor:
    """Class to handle multiple DataFrames with the same filter criteria."""
    
    def __init__(self):
        self.dataframes = {}
        self.results = {}
    
    def add_dataframe(self, name: str, df: pd.DataFrame):
        """Add a DataFrame to the processor."""
        self.dataframes[name] = df
    
    def load_from_files(self, file_paths: List[str]) -> None:
        """Load multiple DataFrames from file paths."""
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
            
            try:
                df = load_dataframe(file_path)
                # Use filename without extension as the key
                name = Path(file_path).stem
                self.add_dataframe(name, df)
                logger.info(f"Loaded {name}: {df.shape}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
    
    def apply_filters(self, and_filters: Optional[Dict[str, Dict[str, Any]]] = None, 
                     or_filters: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, pd.DataFrame]:
        """Apply the same filters to all DataFrames."""
        results = {}
        
        for name, df in self.dataframes.items():
            try:
                logger.info(f"Filtering {name}...")
                filter_engine = DataFrameFilter(df)
                filtered_df = filter_engine.filter_dataframe(and_filters, or_filters)
                results[name] = filtered_df
                logger.info(f"  {name}: {len(df)} -> {len(filtered_df)} rows")
            except Exception as e:
                logger.error(f"Error filtering {name}: {str(e)}")
                results[name] = pd.DataFrame()  # Empty DataFrame on error
        
        self.results = results
        return results
    
    def get_combined_results(self, add_source_column: bool = True) -> pd.DataFrame:
        """Combine all filtered results into a single DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        combined_dfs = []
        for name, df in self.results.items():
            if len(df) > 0:
                if add_source_column:
                    df_copy = df.copy()
                    df_copy['_source_file'] = name
                    combined_dfs.append(df_copy)
                else:
                    combined_dfs.append(df)
        
        if combined_dfs:
            return pd.concat(combined_dfs, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def save_individual_results(self, output_dir: str, format: str = 'xlsx') -> None:
        """Save each filtered DataFrame to a separate file."""
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in self.results.items():
            if len(df) > 0:
                output_path = os.path.join(output_dir, f"filtered_{name}.{format}")
                save_dataframe(df, output_path)
            else:
                logger.warning(f"No data to save for {name}")
    
    def get_summary_report(self) -> pd.DataFrame:
        """Generate a summary report of filtering results."""
        summary_data = []
        
        for name in self.dataframes.keys():
            original_count = len(self.dataframes[name])
            filtered_count = len(self.results.get(name, pd.DataFrame()))
            reduction_pct = ((original_count - filtered_count) / original_count * 100) if original_count > 0 else 0
            
            summary_data.append({
                'File': name,
                'Original_Rows': original_count,
                'Filtered_Rows': filtered_count,
                'Rows_Removed': original_count - filtered_count,
                'Reduction_Percent': round(reduction_pct, 2)
            })
        
        return pd.DataFrame(summary_data)

def parse_filter_string(filter_string: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse filter string into structured format.
    
    Format: 'column1:operator1:value1,column2:operator2:value2'
    If no operator specified, defaults to '=='
    For null checks: 'column:isnull' or 'column:isnotnull' (no value needed)
    """
    filters = {}
    if not filter_string:
        return filters
    
    for item in filter_string.split(","):
        parts = item.strip().split(":")
        if len(parts) == 2:
            column, value_or_op = parts
            column = column.strip()
            value_or_op = value_or_op.strip()
            
            # Check if it's a null operator
            if value_or_op.lower() in ['isnull', 'isnotnull']:
                filters[column] = {'operator': value_or_op.lower(), 'value': None}
            else:
                # Default to == operator
                filters[column] = {'operator': '==', 'value': value_or_op}
        elif len(parts) == 3:  # column:operator:value
            column, operator, value = parts
            column = column.strip()
            operator = operator.strip()
            
            # For null operators, ignore the value
            if operator.lower() in ['isnull', 'isnotnull']:
                filters[column] = {'operator': operator.lower(), 'value': None}
            else:
                filters[column] = {'operator': operator, 'value': value.strip()}
        else:
            raise ValueError(f"Invalid filter format: {item}. Use 'column:value', 'column:operator:value', or 'column:isnull'")
    
    return filters

def load_dataframe(file_path: str) -> pd.DataFrame:
    """Load DataFrame from various file formats."""
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        else:
            # Try to infer format
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                return pd.read_excel(file_path)
            else:
                return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        raise

def save_dataframe(df: pd.DataFrame, file_path: str) -> None:
    """Save DataFrame to various file formats."""
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df.to_excel(file_path, index=False)
        elif file_path.endswith('.csv'):
            df.to_csv(file_path, index=False)
        elif file_path.endswith('.json'):
            df.to_json(file_path, orient='records', indent=2)
        elif file_path.endswith('.parquet'):
            df.to_parquet(file_path, index=False)
        else:
            # Default to Excel
            df.to_excel(file_path, index=False)
        logger.info(f"Data saved to '{file_path}'")
    except Exception as e:
        logger.error(f"Error saving file {file_path}: {str(e)}")
        raise

def expand_file_patterns(patterns: List[str]) -> List[str]:
    """Expand file patterns (wildcards) to actual file paths."""
    expanded_files = []
    for pattern in patterns:
        if '*' in pattern or '?' in pattern:
            matches = glob.glob(pattern)
            if matches:
                expanded_files.extend(matches)
            else:
                logger.warning(f"No files found matching pattern: {pattern}")
        else:
            expanded_files.append(pattern)
    return expanded_files

def main():
    parser = argparse.ArgumentParser(
        description="Filter multiple DataFrames with the same conditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter single file
  python script.py --input_files data.xlsx --and_filters "Age:>:18"
  
  # Filter multiple specific files
  python script.py --input_files data1.xlsx data2.xlsx data3.csv --and_filters "Status:Active"
  
  # Filter all Excel files in directory
  python script.py --input_files "*.xlsx" --and_filters "Department:Engineering"
  
  # Filter rows where Email is null
  python script.py --input_files "*.xlsx" --and_filters "Email:isnull"
  
  # Filter rows where Phone is not null
  python script.py --input_files "*.csv" --and_filters "Phone:isnotnull"
  
  # Combine null check with other filters
  python script.py --input_files "*.xlsx" --and_filters "Status:Active,Email:isnotnull"
  
  # Filter with pattern and save individually
  python script.py --input_files "sales_*.xlsx" --and_filters "Revenue:>:10000" --save_individual --output_dir results
  
  # Combine all results into one file
  python script.py --input_files "*.csv" --and_filters "Score:>=:80" --combine_results --output_file top_performers.xlsx
  
  # Generate summary report
  python script.py --input_files "*.xlsx" --and_filters "Active:TRUE" --summary_report
        """
    )
    
    parser.add_argument("--input_files", nargs='+', required=True,
                       help="Input files (supports wildcards like *.xlsx, *.csv)")
    parser.add_argument("--and_filters", type=str, 
                       help="AND filters: 'column1:operator1:value1,column2:operator2:value2' or 'column:isnull'")
    parser.add_argument("--or_filters", type=str, 
                       help="OR filters: 'column1:operator1:value1,column2:operator2:value2' or 'column:isnull'")
    
    # Output options
    parser.add_argument("--output_file", type=str, default="filtered_data.xlsx", 
                       help="Output file for combined results")
    parser.add_argument("--output_dir", type=str, default="filtered_results", 
                       help="Directory for individual result files")
    parser.add_argument("--output_format", type=str, default="xlsx", 
                       choices=['xlsx', 'csv', 'json', 'parquet'],
                       help="Output format for individual files")
    
    # Processing options
    parser.add_argument("--combine_results", action="store_true", 
                       help="Combine all filtered results into one file")
    parser.add_argument("--save_individual", action="store_true", 
                       help="Save each filtered DataFrame to a separate file")
    parser.add_argument("--add_source_column", action="store_true", default=True,
                       help="Add source filename column when combining results")
    
    # Display options
    parser.add_argument("--show_info", action="store_true", 
                       help="Show DataFrame info for each file")
    parser.add_argument("--show_sample", action="store_true", 
                       help="Show sample of filtered data")
    parser.add_argument("--summary_report", action="store_true", 
                       help="Generate and display summary report")
    
    args = parser.parse_args()
    
    # Expand file patterns
    file_paths = expand_file_patterns(args.input_files)
    
    if not file_paths:
        logger.error("No input files found")
        sys.exit(1)
    
    logger.info(f"Processing {len(file_paths)} files...")
    
    # Initialize processor
    processor = MultiDataFrameProcessor()
    
    # Load DataFrames
    processor.load_from_files(file_paths)
    
    if not processor.dataframes:
        logger.error("No DataFrames loaded successfully")
        sys.exit(1)
    
    # Show info if requested
    if args.show_info:
        for name, df in processor.dataframes.items():
            print(f"\n=== {name} ===")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Data types:\n{df.dtypes}")
            print(f"Null counts:\n{df.isnull().sum()}")
            print(f"First 3 rows:\n{df.head(3)}")
    
    # Parse filters
    and_filters = parse_filter_string(args.and_filters)
    or_filters = parse_filter_string(args.or_filters)
    
    if not and_filters and not or_filters:
        logger.warning("No filters specified. Output will be identical to input.")
    
    # Apply filters
    try:
        results = processor.apply_filters(and_filters, or_filters)
        
        # Show sample if requested
        if args.show_sample:
            for name, df in results.items():
                if len(df) > 0:
                    print(f"\n=== Sample from {name} ===")
                    print(df.head())
        
        # Generate summary report
        if args.summary_report:
            summary = processor.get_summary_report()
            print("\n=== FILTERING SUMMARY ===")
            print(summary.to_string(index=False))
            print(f"\nTotal original rows: {summary['Original_Rows'].sum()}")
            print(f"Total filtered rows: {summary['Filtered_Rows'].sum()}")
            print(f"Overall reduction: {summary['Rows_Removed'].sum()} rows")
        
        # Save results
        if args.combine_results:
            combined_df = processor.get_combined_results(args.add_source_column)
            if len(combined_df) > 0:
                save_dataframe(combined_df, args.output_file)
                logger.info(f"Combined results: {len(combined_df)} total rows")
            else:
                logger.warning("No data to combine")
        
        if args.save_individual:
            processor.save_individual_results(args.output_dir, args.output_format)
        
        # Default behavior: save combined results if no specific output option chosen
        if not args.combine_results and not args.save_individual:
            combined_df = processor.get_combined_results(args.add_source_column)
            if len(combined_df) > 0:
                save_dataframe(combined_df, args.output_file)
                logger.info(f"Combined results saved to {args.output_file}")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
