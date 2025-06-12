import pandas as pd
import logging
import time
from .numericdata import create_cleaning_pipeline, generate_numeric_cleaning_report
from typing import Dict
import os
from .config import DEFAULT_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main2(input_path: str = DEFAULT_CONFIG["input_file"], output_path: str = DEFAULT_CONFIG["output_file"], config: Dict = None):
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
        report_path = DEFAULT_CONFIG["report_file"]
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