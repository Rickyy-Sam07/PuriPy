# main0cat.py - Automated categorical data cleaning utility
import pandas as pd
import numpy as np
import logging
import sys
import os
import time
import argparse
from typing import List, Dict, Any
from .categoricaldata import SmartCategoricalCleaner, clean_all_categorical_columns, save_cleaning_report_as_text
from .config import config2

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('categoricalapp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def detect_categorical_columns(df: pd.DataFrame, max_unique_ratio: float = 0.5) -> List[str]:
    """
    Automatically detect categorical columns in a dataframe
    
    Args:
        df: Input dataframe
        max_unique_ratio: Maximum ratio of unique values to rows to be considered categorical
        
    Returns:
        List of detected categorical column names
    """
    categorical_cols = []
    num_rows = len(df)
    
    for col in df.columns:
        # Skip columns with high percentage of missing values
        if df[col].isna().mean() > 0.5:
            continue
            
        # Object columns are likely categorical
        if df[col].dtype == 'object':
            unique_ratio = df[col].nunique() / num_rows
            if unique_ratio <= max_unique_ratio:
                categorical_cols.append(col)
                
        # Numeric columns with low cardinality are likely categorical
        elif pd.api.types.is_numeric_dtype(df[col]):
            unique_count = df[col].nunique()
            if unique_count < 15 and unique_count / num_rows <= 0.1:
                categorical_cols.append(col)
    
    return categorical_cols

def preview_data(df: pd.DataFrame) -> None:
    """Print an informative preview of the dataset"""
    print("\n===== DATA PREVIEW =====")
    print(f"Dataset shape: {df.shape} (rows, columns)")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nColumn information:")
    for col in df.columns:
        col_type = df[col].dtypesv
        unique_count = df[col].nunique()
        missing = df[col].isna().sum()
        missing_percent = missing / len(df) * 100
        print(f"- {col}: type={col_type}, unique={unique_count}, missing={missing} ({missing_percent:.1f}%)")

def main0(put_path: str = config2["INPUT_FILE"], 
         output_path: str = config2["OUTPUT_FILE"],
         target_column: str = config2["TARGET_COLUMN"],
         columns: List[str] = config2["COLUMNS_TO_CLEAN"],
         exclude_columns: List[str] = config2["EXCLUDE_COLUMNS"],
         file_path: str = config2["FILE_PATH"],
         explore_only: bool = config2["EXPLORE_ONLY"],
         fix_typos: bool = config2["FIX_TYPOS"],
         group_rare: bool = config2["GROUP_RARE"],
         rare_threshold: float = config2["RARE_THRESHOLD"],
         similarity_threshold: float = config2["SIMILARITY_THRESHOLD"],
         memory_efficient: bool = config2["MEMORY_EFFICIENT"],
         parallel_jobs: int = config2["PARALLEL_JOBS"]):
    """
    main0 workflow for categorical data cleaning with automatic detection
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned data
        target_column: Target column for supervised learning (if any)
        columns: Specific columns to clean (if None, auto-detect)
        exclude_columns: Columns to exclude from cleaning
        file_path: Path to save the cleaning report
        explore_only: If True, only explore data without cleaning
        fix_typos: Whether to fix typos
        group_rare: Whether to group rare categories
        rare_threshold: Threshold for rare category grouping
        similarity_threshold: Threshold for fuzzy matching
        memory_efficient: Whether to operate in memory-efficient mode
        parallel_jobs: Number of parallel jobs
    """
    start_time = time.time()
    
    try:
        # Load data
        logger.info(f"Loading dataset from {input_path}")
        df = pd.read_csv(input_path)
        
        if df.empty:
            print("Warning: The dataset is empty.")
            return
            
        # Show data preview to help user understand the dataset
        preview_data(df)
        
        # Auto-detect categorical columns if not specified
        if not columns:
            categorical_columns = detect_categorical_columns(df)
            # Remove excluded columns
            categorical_columns = [col for col in categorical_columns if col not in exclude_columns]
            
            if not categorical_columns:
                print("No categorical columns detected in the dataset.")
                return
            print(f"\nDetected categorical columns: {', '.join(categorical_columns)}")
        else:
            # Validate that specified columns exist
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: Columns not found in dataset: {', '.join(missing_cols)}")
                
            # Filter out columns that don't exist and excluded columns
            categorical_columns = [col for col in columns if col in df.columns and col not in exclude_columns]
            if not categorical_columns:
                print("None of the specified columns exist in the dataset.")
                return
                
        # If explore_only mode, exit here
        if explore_only:
            print("\nExploration complete. No cleaning performed.")
            return
        
        # Initialize cleaner with the dataframe
        print("\n===== CLEANING PROCESS =====")
        
        # Check if we should use the batch method or individual column method
        if len(categorical_columns) > 3 and target_column is None:
            # Use batch cleaning for efficiency with multiple columns
            print(f"Cleaning {len(categorical_columns)} columns in batch mode...")
            df = clean_all_categorical_columns(
                df, 
                target_column=target_column,
                excluded_columns=exclude_columns,
                n_jobs=parallel_jobs,
                memory_efficient=memory_efficient,
                fix_typos=fix_typos,
                group_rare=group_rare,
                rare_threshold=rare_threshold,
                similarity_threshold=similarity_threshold
            )
            
            print(f"Batch cleaning complete.")
        else:
            # Use individual column cleaning for more control and detailed progress
            cleaner = SmartCategoricalCleaner(
                df, 
                target_column=target_column,
                n_jobs=parallel_jobs,
                memory_efficient=memory_efficient
            )
            
            # Clean each detected categorical column
            for col in categorical_columns:
                print(f"\nCleaning column: {col}")
                df = cleaner.auto_clean(
                    col,
                    fix_typos=fix_typos,
                    group_rare=group_rare,
                    rare_threshold=rare_threshold,
                    similarity_threshold=similarity_threshold
                )
            save_cleaning_report_as_text(cleaner, file_path)
            print(f"Cleaning report saved to: {file_path}")
        
        # Save results
        df.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to: {output_path}")
        
        # Generate and display cleaning report if individual column cleaning was used
        if 'cleaner' in locals():
            report = cleaner.get_cleaning_report()
            print("\n===== CLEANING REPORT =====")
            for col, details in report.items():
                print(f"\nColumn: {col}")
                print(f"  Actions performed:")
                for action in details.get('actions_performed', []):
                    print(f"    - {action}")
                
                before_stats = details.get('before_stats', {})
                unique_values = before_stats.get('unique_values', 'N/A')
                missing_values = before_stats.get('missing_values', 'N/A')
                print(f"  Before: {unique_values} unique values, {missing_values} missing")
                
                after_stats = details.get('after_stats', {})
                unique_after = after_stats.get('nunique', 'N/A')  
                missing_after = after_stats.get('missing', 'N/A')
                print(f"  After: {unique_after} unique values, {missing_after} missing")

        # Print sample of cleaned data
        print("\n===== CLEANED DATA SAMPLE =====")
        print(df.head())

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
    except pd.errors.EmptyDataError:
        print(f"Error: Input file is empty: {input_path}")
    except pd.errors.ParserError:
        print(f"Error: Unable to parse {input_path}, check if it's a valid CSV file")
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.exception("Unexpected error occurred")
    
    finally:
        # Calculate and show execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    # Check if command-line arguments were provided (advanced usage)
    if len(sys.argv) > 1:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Smart Categorical Data Cleaner')
        parser.add_argument('--input', help='Input CSV file path', default=config2["INPUT_FILE"])
        parser.add_argument('--output', help='Output CSV file path', default=config2["OUTPUT_FILE"])
        parser.add_argument('--target', help='Target column name for supervised learning', default=config2["TARGET_COLUMN"])
        parser.add_argument('--columns', help='Comma-separated list of columns to clean (if omitted, auto-detect)', default=None)
        parser.add_argument('--exclude', help='Comma-separated list of columns to exclude', default=None)
        parser.add_argument('--explore', help='Only explore data without cleaning', action='store_true', default=config2["EXPLORE_ONLY"])
        parser.add_argument('--no-typos', help='Disable typo correction', action='store_false', dest='fix_typos', default=config2["FIX_TYPOS"])
        parser.add_argument('--no-rare', help='Disable rare category grouping', action='store_false', dest='group_rare', default=config2["GROUP_RARE"])
        parser.add_argument('--rare-threshold', help='Threshold for rare category grouping (0-1)', type=float, default=config2["RARE_THRESHOLD"])
        parser.add_argument('--similarity', help='Similarity threshold for fuzzy matching (0-100)', type=float, default=config2["SIMILARITY_THRESHOLD"])
        parser.add_argument('--memory-efficient', help='Use memory-efficient mode', action='store_true', default=config2["MEMORY_EFFICIENT"])
        parser.add_argument('--jobs', help='Number of parallel jobs (-1 for all cores)', type=int, default=config2["PARALLEL_JOBS"])
        parser.add_argument('--file', help='Path to save the cleaning report', default=config2["FILE_PATH"])
        args = parser.parse_args()
        
        # Process column lists if provided
        column_list = None
        if args.columns:
            column_list = [col.strip() for col in args.columns.split(',')]
            
        exclude_list = []
        if args.exclude:
            exclude_list = [col.strip() for col in args.exclude.split(',')]
        
        main0(
            input_path=args.input, 
            output_path=args.output,
            target_column=args.target,
            columns=column_list,
            exclude_columns=exclude_list,
            explore_only=args.explore,
            fix_typos=args.fix_typos,
            group_rare=args.group_rare,
            rare_threshold=args.rare_threshold,
            similarity_threshold=args.similarity,
            memory_efficient=args.memory_efficient,
            parallel_jobs=args.jobs,
            file_path=args.file
        )
    else:
        main0()
