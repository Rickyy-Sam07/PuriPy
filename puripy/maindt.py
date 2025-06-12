# main3dt.py - Automated date and time data cleaning utility
import pandas as pd
import numpy as np
import logging
import sys
import os
import time
import argparse
from typing import List, Dict, Tuple, Optional, Any
from .dt import DateTimeCleaner, clean_datetime_columns, save_datetime_cleaning_report_as_text
from .config import config3


# config3ure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('datetime_cleaner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def preview_data(df: pd.DataFrame) -> None:
    """Print an informative preview of the dataset"""
    print("\n===== DATA PREVIEW =====")
    print(f"Dataset shape: {df.shape} (rows, columns)")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nColumn information:")
    for col in df.columns:
        col_type = df[col].dtype
        unique_count = df[col].nunique()
        missing = df[col].isna().sum()
        missing_percent = missing / len(df) * 100
        print(f"- {col}: type={col_type}, unique={unique_count}, missing={missing} ({missing_percent:.1f}%)")

def main3(input_path: str = config3["INPUT_FILE"],
         output_path: str = config3["OUTPUT_FILE"],
         date_columns: List[str] = config3["DATE_COLUMNS"],
         start_end_pairs: List[Tuple[str, str]] = config3["START_END_PAIRS"],
         parse_dates: bool = config3["PARSE_DATES"],
         impute_missing: bool = config3["IMPUTE_MISSING"],
         imputation_method: str = config3["IMPUTATION_METHOD"],
         seasonal_period: int = config3["SEASONAL_PERIOD"], 
         standardize_timezone: bool = config3["STANDARDIZE_TIMEZONE"],
         target_timezone: str = config3["TARGET_TIMEZONE"],
         extract_features: bool = config3["EXTRACT_FEATURES"],
         calendar_features: List[str] = config3["CALENDAR_FEATURES"],
         consistency_action: str = config3["CONSISTENCY_ACTION"],
         fiscal_year_start: int = config3["FISCAL_YEAR_START"],
         memory_efficient: bool = config3["MEMORY_EFFICIENT"]) -> None:
    """
    main3 workflow for date and time data cleaning
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned data
        date_columns: List of datetime columns (None for auto-detect)
        start_end_pairs: List of (start_date, end_date) column pairs to validate
        parse_dates: Whether to parse string dates to datetime
        impute_missing: Whether to impute missing values
        imputation_method: Method for imputation
        seasonal_period: Period for seasonal imputation
        standardize_timezone: Whether to standardize timezone
        target_timezone: Target timezone for standardization
        extract_features: Whether to extract calendar features
        calendar_features: List of calendar features to extract
        consistency_action: How to handle inconsistent dates
        fiscal_year_start: Starting month of fiscal year (1-12)
        memory_efficient: Whether to operate in memory-efficient mode
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
        
        # Convert start_end_pairs if provided
        validated_pairs = []
        if start_end_pairs:
            for start_col, end_col in start_end_pairs:
                if start_col in df.columns and end_col in df.columns:
                    validated_pairs.append((start_col, end_col))
                else:
                    missing_cols = [col for col in [start_col, end_col] if col not in df.columns]
                    print(f"Warning: Columns not found for date range validation: {', '.join(missing_cols)}")
        
        # Process datetime columns
        print("\n===== CLEANING PROCESS =====")
        print("Cleaning date and time columns...")
        
        # Use the DateTimeCleaner for more control
        cleaner = DateTimeCleaner(df, memory_efficient=memory_efficient)
        
        # Auto-detect datetime columns if not specified
        if not date_columns:
            detected_columns = cleaner.auto_detect_datetime_columns()
            if not detected_columns:
                print("No datetime columns detected. Please check your data or specify columns manually.")
                return
            print(f"Detected {len(detected_columns)} datetime columns: {', '.join(detected_columns)}")
            date_columns = detected_columns
        else:
            # Filter out columns that don't exist
            date_columns = [col for col in date_columns if col in df.columns]
            if not date_columns:
                print("None of the specified date columns exist in the dataset.")
                return
        
        # Process each datetime column
        for col in date_columns:
            print(f"\nProcessing column: {col}")
            
            if parse_dates:
                print(f"  - Parsing dates...")
                cleaner.parse_datetime(col)
                
            if impute_missing and df[col].isna().any():
                print(f"  - Imputing missing values using {imputation_method}...")
                if imputation_method == 'seasonal':
                    cleaner.impute_missing_dates(col, method=imputation_method, seasonal_period=seasonal_period)
                else:
                    cleaner.impute_missing_dates(col, method=imputation_method)
                    
            if standardize_timezone:
                print(f"  - Standardizing timezone to {target_timezone}...")
                cleaner.standardize_timezone(col, target_timezone=target_timezone)
                
            if extract_features:
                print(f"  - Extracting calendar features...")
                cleaner.extract_calendar_features(col, features=calendar_features, fiscal_year_start=fiscal_year_start)
        
        # Validate temporal consistency for start-end pairs
        if validated_pairs:
            print("\nValidating date range consistency...")
            for start_col, end_col in validated_pairs:
                print(f"  - Checking {start_col} ≤ {end_col}...")
                cleaner.validate_temporal_consistency(start_col, end_col, action=consistency_action)
        
        # Get the cleaned dataframe
        df_cleaned = cleaner.df
        
        # Generate detailed report
        report_path = config3["REPORT_FILE"]
        save_datetime_cleaning_report_as_text(cleaner, file_path=report_path)
        print(f"\nDetailed cleaning report saved to: {report_path}")
        
        # Save results
        df_cleaned.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to: {output_path}")
        
        # Display report summary
        print("\n===== CLEANING SUMMARY =====")
        for col, details in cleaner.get_report().items():
            print(f"\nColumn: {col}")
            
            # Display actions
            actions = details.get('actions', [])
            if actions:
                print(f"  Actions performed: {len(actions)}")
                
            # Display created features
            created_features = details.get('created_features', [])
            if created_features:
                print(f"  Features created: {len(created_features)}")
        
        # Print sample of cleaned data
        print("\n===== CLEANED DATA SAMPLE =====")
        sample_cols = min(10, len(df_cleaned.columns))
        print(df_cleaned.head()[df_cleaned.columns[:sample_cols]])
        if sample_cols < len(df_cleaned.columns):
            print(f"... and {len(df_cleaned.columns) - sample_cols} more columns")

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
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
        parser = argparse.ArgumentParser(description='Date and Time Data Cleaner')
        parser.add_argument('--input', help='Input CSV file path', default=config3["INPUT_FILE"])
        parser.add_argument('--output', help='Output CSV file path', default=config3["OUTPUT_FILE"])
        
        parser.add_argument('--columns', help='Comma-separated list of datetime columns (if omitted, auto-detect)', default=None)
        parser.add_argument('--pairs', help='Comma-separated list of start-end column pairs, format: "start1:end1,start2:end2"', default=None)
        
        parser.add_argument('--no-parse', help='Disable date parsing', action='store_false', dest='parse_dates', default=config3["PARSE_DATES"])
        parser.add_argument('--no-impute', help='Disable missing value imputation', action='store_false', dest='impute_missing', default=config3["IMPUTE_MISSING"])
        parser.add_argument('--imputation', help='Imputation method', choices=['linear', 'forward', 'backward', 'seasonal', 'mode'], default=config3["IMPUTATION_METHOD"])
        parser.add_argument('--period', help='Period for seasonal imputation', type=int, default=config3["SEASONAL_PERIOD"])
        
        parser.add_argument('--timezone', help='Standardize to specific timezone', action='store_true', dest='standardize_timezone', default=config3["STANDARDIZE_TIMEZONE"])
        parser.add_argument('--target-tz', help='Target timezone', default=config3["TARGET_TIMEZONE"])
        
        parser.add_argument('--no-features', help='Disable feature extraction', action='store_false', dest='extract_features', default=config3["EXTRACT_FEATURES"])
        parser.add_argument('--features', help='Comma-separated list of calendar features to extract', default=None)
        
        parser.add_argument('--consistency', help='How to handle inconsistent dates', choices=['flag', 'swap', 'drop', 'truncate'], default=config3["CONSISTENCY_ACTION"])
        parser.add_argument('--fiscal-start', help='Fiscal year starting month (1-12)', type=int, default=config3["FISCAL_YEAR_START"])
        parser.add_argument('--memory-efficient', help='Use memory-efficient mode', action='store_true', default=config3["MEMORY_EFFICIENT"])
        
        args = parser.parse_args()
        
        # Process column lists if provided
        date_cols = None
        if args.columns:
            date_cols = [col.strip() for col in args.columns.split(',')]
            
        # Process start-end pairs if provided
        pairs = []
        if args.pairs:
            for pair in args.pairs.split(','):
                if ':' in pair:
                    start, end = pair.split(':')
                    pairs.append((start.strip(), end.strip()))
                    
        # Process calendar features if provided
        cal_features = config3["CALENDAR_FEATURES"]
        if args.features:
            cal_features = [f.strip() for f in args.features.split(',')]
        
        main3(
            input_path=args.input, 
            output_path=args.output,
            date_columns=date_cols,
            start_end_pairs=pairs,
            parse_dates=args.parse_dates,
            impute_missing=args.impute_missing,
            imputation_method=args.imputation,
            seasonal_period=args.period,
            standardize_timezone=args.standardize_timezone,
            target_timezone=args.target_tz,
            extract_features=args.extract_features,
            calendar_features=cal_features,
            consistency_action=args.consistency,
            fiscal_year_start=args.fiscal_start,
            memory_efficient=args.memory_efficient
        )
    else:
        # Use the default config3 from the top of the file
        main3()