import pandas as pd
import argparse
import sys
from typing import Dict, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataFrameFilter:
    """A class to handle DataFrame filtering operations with various operators."""
    
    VALID_OPERATORS = {
        'string': ['==', '!=', 'contains', 'startswith', 'endswith', 'notcontains', 'icontains'],
        'numeric': ['==', '!=', '<', '<=', '>', '>='],
        'common': ['==', '!=']
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
            value: Value to compare against
            
        Returns:
            Filtered DataFrame
        """
        self._validate_column(column_name)
        
        # Convert value to appropriate type
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
        result_df = self.dataframe.copy()
        
        # Apply AND filters sequentially
        if and_filters:
            for column, filter_config in and_filters.items():
                operator = filter_config.get('operator', '==')
                value = filter_config['value']
                temp_filter = DataFrameFilter(result_df)
                result_df = temp_filter.single_filter(column, operator, value)
        
        # Apply OR filters
        if or_filters:
            or_conditions = []
            for column, filter_config in or_filters.items():
                operator = filter_config.get('operator', '==')
                value = filter_config['value']
                temp_filter = DataFrameFilter(self.dataframe)
                filtered_df = temp_filter.single_filter(column, operator, value)
                or_conditions.append(filtered_df.index)
            
            if or_conditions:
                # Combine all OR condition indices
                combined_indices = set()
                for indices in or_conditions:
                    combined_indices.update(indices)
                
                # If we have AND filters, intersect with OR results
                if and_filters:
                    result_df = result_df[result_df.index.isin(combined_indices)]
                else:
                    result_df = self.dataframe[self.dataframe.index.isin(combined_indices)]
        
        return result_df

def parse_filter_string(filter_string: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse filter string into structured format.
    
    Format: 'column1:operator1:value1,column2:operator2:value2'
    If no operator specified, defaults to '=='
    """
    filters = {}
    if not filter_string:
        return filters
    
    for item in filter_string.split(","):
        parts = item.strip().split(":")
        if len(parts) == 2:  # column:value (default to ==)
            column, value = parts
            filters[column.strip()] = {'operator': '==', 'value': value.strip()}
        elif len(parts) == 3:  # column:operator:value
            column, operator, value = parts
            filters[column.strip()] = {'operator': operator.strip(), 'value': value.strip()}
        else:
            raise ValueError(f"Invalid filter format: {item}. Use 'column:value' or 'column:operator:value'")
    
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
        sys.exit(1)

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
        logger.info(f"Filtered data saved to '{file_path}'")
    except Exception as e:
        logger.error(f"Error saving file {file_path}: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Filter a DataFrame based on specified conditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple equality filter
  python script.py --input_file data.xlsx --and_filters "Name:John,Age:25"
  
  # Using operators
  python script.py --input_file data.xlsx --and_filters "Age:>:18,Score:>=:80"
  
  # String operations
  python script.py --input_file data.xlsx --and_filters "Name:contains:John,Email:endswith:.com"
  
  # Combining AND and OR filters
  python script.py --input_file data.xlsx --and_filters "Age:>:18" --or_filters "City:New York,City:Los Angeles"
        """
    )
    
    parser.add_argument("--input_file", type=str, default="BASE.xlsx", 
                       help="Input file (supports .xlsx, .csv, .json, .parquet)")
    parser.add_argument("--and_filters", type=str, 
                       help="AND filters: 'column1:operator1:value1,column2:operator2:value2'")
    parser.add_argument("--or_filters", type=str, 
                       help="OR filters: 'column1:operator1:value1,column2:operator2:value2'")
    parser.add_argument("--output_file", type=str, default="filtered_data.xlsx", 
                       help="Output file (supports .xlsx, .csv, .json, .parquet)")
    parser.add_argument("--show_info", action="store_true", 
                       help="Show DataFrame info before filtering")
    parser.add_argument("--show_sample", action="store_true", 
                       help="Show sample of filtered data")
    
    args = parser.parse_args()
    
    # Load DataFrame
    logger.info(f"Loading data from {args.input_file}")
    df = load_dataframe(args.input_file)
    
    if args.show_info:
        print(f"\nDataFrame Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print(f"\nFirst 5 rows:")
        print(df.head())
    
    # Parse filters
    and_filters = parse_filter_string(args.and_filters)
    or_filters = parse_filter_string(args.or_filters)
    
    if not and_filters and not or_filters:
        logger.warning("No filters specified. Output will be identical to input.")
    
    # Apply filters
    filter_engine = DataFrameFilter(df)
    
    try:
        filtered_df = filter_engine.filter_dataframe(and_filters, or_filters)
        
        logger.info(f"Filtering complete. Rows: {len(df)} -> {len(filtered_df)}")
        
        if args.show_sample and len(filtered_df) > 0:
            print(f"\nSample of filtered data:")
            print(filtered_df.head())
        elif len(filtered_df) == 0:
            logger.warning("No rows match the specified filters.")
        
        # Save filtered data
        save_dataframe(filtered_df, args.output_file)
        
    except Exception as e:
        logger.error(f"Error during filtering: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()