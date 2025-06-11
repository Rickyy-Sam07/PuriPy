import pandas as pd
import logging
import time
from numericdata import create_cleaning_pipeline, generate_numeric_cleaning_report
from typing import Dict
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# Data Cleaning Configuration
# --------------------------
# This configuration controls how the data cleaning pipeline processes your dataset.
# Modify these settings based on your specific dataset requirements.

DEFAULT_CONFIG = {
    # Type Conversion Settings
    # -----------------------
    # Specify which columns should be converted to numeric types
    # For non-numeric datasets, set this to [] or remove columns that aren't numeric
    'type_conversion': {
        'numeric_cols': ['Sales_Before', 'Sales_After', 'Customer_Satisfaction_Before', 'Customer_Satisfaction_After']
    },
    
    # Missing Value Handling
    # ---------------------
    # strategy: how to fill missing values ('mean', 'median', 'mode')
    # threshold: maximum ratio of missing values allowed (0.0 to 1.0)
    'missing_values': {
        'strategy': 'mean',  # Options: 'mean', 'median', 'mode'
        'threshold': 0.5     # Columns with >40% missing values will be flagged
    },
    
    # Data Constraints & Validation
    # ----------------------------
    # Define valid ranges/rules for each column using lambda functions
    # correction: how to replace invalid values ('median', 'mean', 'mode')
    'data_errors': {
        'constraints': {
            'Sales_Before': lambda x: (x >= 50) & (x <= 500),  # Valid sales range
            'Sales_After': lambda x: (x >= 50) & (x <= 700),  # Valid sales range
            'Customer_Satisfaction_Before': lambda x: (x >= 0) & (x <= 100),  # Percentage-based score
            'Customer_Satisfaction_After': lambda x: (x >= 0) & (x <= 100)  # Percentage-based score
        },
        'correction': 'median'  # Use median of valid values to replace invalid ones
    },
    
    # Outlier Detection & Handling
    # --------------------------
    # method: technique to detect outliers ('iqr', 'zscore')
    # action: how to handle outliers ('cap', 'remove')
    # columns: specific columns to check for outliers
    'outliers': {
        'method': 'iqr',  # Interquartile Range method (Q1-1.5*IQR to Q3+1.5*IQR)
        'action': 'cap',  # Cap values at the boundaries instead of removing rows
        'columns': ['Sales_Before', 'Sales_After', 'Customer_Satisfaction_Before', 'Customer_Satisfaction_After']  # Columns to check
    },
    
    # Duplicate Handling
    # -----------------
    # subset: columns to consider when identifying duplicates (None = all columns)
    # keep: which occurrence to keep ('first', 'last', False)
    'duplicates': {
        'subset': None,  # Consider all columns when identifying duplicates
        'keep': 'first'  # Keep the first occurrence and remove others
    },
    
    # Numeric Precision
    # ----------------
    # Control decimal places for each column (0 = integer, >0 = decimal places)
    'precision': {
        'Sales_Before': 2,      # Two decimal places for currency
        'Sales_After': 2,       # Two decimal places for currency
        'Customer_Satisfaction_Before': 1,  # One decimal place for satisfaction scores
        'Customer_Satisfaction_After': 1    # One decimal place for satisfaction scores
    }
}




def main2(input_path: str = "data.csv", output_path: str = "cleaned_output.csv", config: Dict = None):
    """
    Clean numeric data from a CSV file based on the provided configuration.
    
    Parameters:
        input_path (str): Path to input CSV file
        output_path (str): Path for output cleaned CSV file
        config (Dict, optional): Custom configuration (defaults to DEFAULT_CONFIG)
    """
    total_start = time.time()
    try:
        if config is None:
            config = DEFAULT_CONFIG
            logger.info("Using default cleaning configuration")
        
        logger.info(f"Loading dataset from {input_path}")
        df = pd.read_csv(input_path)
        original_df = df.copy()
        
        df = df.replace(['not_available', 'N/A', 'na', 'unknown'], pd.NA)
        
        logger.info(f"Initial shape: {df.shape}")
        
        pipeline = create_cleaning_pipeline(config)
        cleaned_df = pipeline(df)
        
        cleaned_df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to {output_path}")
        
        # Changed to use fixed filename as requested
        report_path = "textreport.txt"
        generate_numeric_cleaning_report(
            original_df=original_df,
            cleaned_df=cleaned_df,
            config=config,
            file_path=report_path
        )
        logger.info(f"Cleaning report generated at {report_path}")
        
        print("\n=== CLEANING RESULTS ===")
        print(cleaned_df.head())
        print(f"\nDetailed cleaning report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise
    finally:
        total_time = time.time() - total_start
        logger.info(f"\n{'='*40}\nTotal execution time: {total_time:.2f} seconds\n{'='*40}")

if __name__ == "__main__":
    main2()