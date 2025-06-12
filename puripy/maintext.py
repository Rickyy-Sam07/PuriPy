# main.py - Text data cleaning utility
import pandas as pd
import time
import os
from .textcleaner import create_text_cleaner, clean_dataframe, auto_detect_text_column, save_text_cleaning_report_as_txt
import logging
from .config import config

# Configure logging to match textcleaner's setup
logger = logging.getLogger(__name__)

def main():
    # Start timing
    start_time = time.time()
    
    try:
        # Load raw data
        load_start = time.time()
        input_file = config.get('input_file', r"testdata\test.csv")
        output_file = config.get('output_file', 'output.csv')
        report_file = config.get('report_file', 'text_cleaning_report.txt')
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        df = pd.read_csv(input_file)
        logger.info(f"Data loaded in {time.time() - load_start:.2f}s")
        
        # Get text column - use configured value or auto-detect
        original_text_col = config.get('text_column')
        if not original_text_col:
            original_text_col = auto_detect_text_column(df)
            logger.info(f"Auto-detected text column: {original_text_col}")
        else:
            logger.info(f"Using configured text column: {original_text_col}")
        
        # Create cleaner
        cleaner = create_text_cleaner(config)
        
        # Clean data
        clean_start = time.time()
        cleaned_df = clean_dataframe(
            df,
            cleaner,
            text_column=original_text_col,
            output_column='cleaned_text',
            parallel=True,
            verbose=True
        )
        logger.info(f"Cleaning completed in {time.time() - clean_start:.2f}s")
        
        # Generate text cleaning report
        report_start = time.time()
        save_text_cleaning_report_as_txt(
            df=cleaned_df,
            text_column=original_text_col,
            output_column='cleaned_text',
            config=config,
            file_path=report_file,
            include_samples=5
        )
        logger.info(f"Report generated in {time.time() - report_start:.2f}s")
        
        # Replace original text column
        if 'cleaned_text' in cleaned_df.columns:
            cleaned_df[original_text_col] = cleaned_df['cleaned_text']
            cleaned_df.drop(columns=['cleaned_text'], inplace=True)
        
        # Save results
        save_start = time.time()
        cleaned_df.to_csv(output_file, index=False)
        logger.info(f"Data saved to {output_file} in {time.time() - save_start:.2f}s")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise
    
    finally:
        # Calculate total time
        total_time = time.time() - start_time
        logger.info(f"\n{'='*40}\nTotal execution time: {total_time:.2f} seconds\n{'='*40}")

if __name__ == "__main__":
    main()
