# mainnum.py
import pandas as pd
import logging
import time
from numericdata import create_cleaning_pipeline
from typing import Dict

# Configure logging
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

"""
CONFIGURATION ADAPTATION GUIDE
-----------------------------

How to adapt the configuration for different datasets:

1. NUMERIC COLUMNS:
   - Update 'numeric_cols' to include only the columns that should be numeric
   - Example for financial data:
     'numeric_cols': ['price', 'quantity', 'total', 'discount']

2. CONSTRAINTS:
   - Define reasonable value ranges for each numeric column based on domain knowledge
   - Examples:
     - Product data: 'price': lambda x: (x >= 0) & (x <= 10000)
     - Health data: 'temperature': lambda x: (x >= 35) & (x <= 42)
     - Web analytics: 'session_time': lambda x: (x >= 0) & (x <= 3600)

3. OUTLIER HANDLING:
   - Choose method based on data distribution:
     - 'iqr': Good for skewed data (default)
     - 'zscore': Better for normally distributed data
   - Choose action based on business requirements:
     - 'cap': Better for keeping all data but limiting extreme values
     - 'remove': Better for training models where outliers are problematic

4. PRECISION:
   - Set decimal precision based on measurement accuracy and reporting needs
   - Examples:
     - Financial: 'amount': 2 (dollars and cents)
     - Scientific: 'measurement': 3 (three decimal precision)
     - Integer IDs: 'user_id': 0 (whole numbers)

5. MISSING VALUES:
   - Choose strategy based on data characteristics:
     - 'mean': Good for normally distributed data
     - 'median': Better for skewed data
     - 'mode': Better for categorical or discrete data

Example config for financial dataset:
```python
FINANCIAL_CONFIG = {
    'type_conversion': {
        'numeric_cols': ['price', 'quantity', 'discount', 'total']
    },
    'missing_values': {
        'strategy': 'median',  # Financial data often has outliers
        'threshold': 0.3       # Stricter threshold for financial data
    },
    'data_errors': {
        'constraints': {
            'price': lambda x: (x >= 0) & (x <= 10000),
            'quantity': lambda x: (x >= 1) & (x <= 1000),
            'discount': lambda x: (x >= 0) & (x <= 0.5),
        },
        'correction': 'median'
    },
    'outliers': {
        'method': 'iqr',
        'action': 'cap',
        'columns': ['price', 'quantity', 'total']
    },
    'precision': {
        'price': 2,       # Dollars and cents
        'quantity': 0,    # Whole units
        'discount': 2,    # Percentage with 2 decimals
        'total': 2        # Dollars and cents
    }
}
```
"""

def main(input_path: str = "data.csv", output_path: str = "cleaned_output.csv", config: Dict = None):
    """
    Clean numeric data from a CSV file based on the provided configuration.
    
    Parameters:
        input_path (str): Path to input CSV file
        output_path (str): Path for output cleaned CSV file
        config (Dict, optional): Custom configuration (defaults to DEFAULT_CONFIG)
    """
    total_start = time.time()
    try:
        # Use default config if none provided
        if config is None:
            config = DEFAULT_CONFIG
            logger.info("Using default cleaning configuration")
        
        # Load dataset
        logger.info(f"Loading dataset from {input_path}")
        df = pd.read_csv(input_path)
        
        # Replace 'not_available' with NaN
        df = df.replace(['not_available', 'N/A', 'na', 'unknown'], pd.NA)
        
        logger.info(f"Initial shape: {df.shape}")
        
        # Apply cleaning pipeline
        pipeline = create_cleaning_pipeline(config)
        cleaned_df = pipeline(df)
        
        # Save results
        cleaned_df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to {output_path}")
        
        print("\n=== CLEANING RESULTS ===")
        print(cleaned_df.head())
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise
    finally:
        total_time = time.time() - total_start
        logger.info(f"\n{'='*40}\nTotal execution time: {total_time:.2f} seconds\n{'='*40}")

# Example of running with a custom configuration
if __name__ == "__main__":
    # Standard usage with default config
    main()
    
    # Uncomment to use a custom config
    # CUSTOM_CONFIG = {
    #     'type_conversion': {'numeric_cols': ['age', 'salary']},
    #     'missing_values': {'strategy': 'median', 'threshold': 0.7},
    #     'data_errors': {'constraints': {'age': lambda x: x > 0}, 'correction': 'median'},
    #     'outliers': {'method': 'iqr', 'action': 'cap', 'columns': ['salary']},
    #     'duplicates': {'subset': None, 'keep': 'first'},
    #     'precision': {'age': 0, 'salary': 0}
    # }
    # main(input_path="custom_data.csv", output_path="custom_cleaned.csv", config=CUSTOM_CONFIG)
