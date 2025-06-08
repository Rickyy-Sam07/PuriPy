# main.py
import pandas as pd
import time
from textcleaner import create_text_cleaner, clean_dataframe, auto_detect_text_column
import logging

# Configure logging to match textcleaner's setup
logger = logging.getLogger(__name__)

config = {
    'lowercase': True,              # Convert all text to lowercase
    'remove_punctuation': True,     # Remove all punctuation marks
    'remove_stopwords': True,       # Remove common stopwords (e.g., "the", "is", "and")
    'remove_urls': True,            # Remove URLs and web addresses
    'remove_html': True,            # Remove HTML tags from text
    'remove_emojis': True,          # Remove emojis and special symbols
    'remove_numbers': True,         # Remove all numeric digits
    'expand_contractions': True,    # Expand contractions (e.g., "don't" -> "do not")
    'spelling_correction': True,    # Correct spelling mistakes in words
    'lemmatize': True,              # Reduce words to their base form (e.g., "running" -> "run")
    'stem': False,                  # Reduce words to their root form (e.g., "running" -> "run"); set True to enable
    'tokenize': 'word',             # Tokenize text into words ('word'), sentences ('sentence'), or None for no tokenization
    'ngram_range': (1, 1),          # Generate n-grams; (1,1) for unigrams only, (1,2) for unigrams and bigrams, etc.
    'profanity_filter': False,      # Remove or mask profane words; set True to enable
    'language': 'english',          # Language for stopwords, lemmatization, and spell checking
    'custom_stopwords': None,       # List of additional stopwords to remove (e.g., ['foo', 'bar']); None for default
    'custom_profanity': None        # List of additional profane words to filter; None for default
}


def main():
    # Start timing
    start_time = time.time()
    
    try:
        # Load raw data
        load_start = time.time()
        df = pd.read_csv('sample.csv')
        logger.info(f"Data loaded in {time.time() - load_start:.2f}s")
        
        # Create cleaner
        cleaner = create_text_cleaner(config)
        
        # Clean data
        clean_start = time.time()
        cleaned_df = clean_dataframe(
            df,
            cleaner,
            parallel=True,
            verbose=True
        )
        logger.info(f"Cleaning completed in {time.time() - clean_start:.2f}s")
        
        # Replace original text column
        original_text_col = auto_detect_text_column(df)
        if 'cleaned_text' in cleaned_df.columns:
            cleaned_df[original_text_col] = cleaned_df['cleaned_text']
            cleaned_df.drop(columns=['cleaned_text'], inplace=True)
        
        # Save results
        save_start = time.time()
        cleaned_df.to_csv('output.csv', index=False)
        logger.info(f"Data saved in {time.time() - save_start:.2f}s")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise
    
    finally:
        # Calculate total time
        total_time = time.time() - start_time
        logger.info(f"\n{'='*40}\nTotal execution time: {total_time:.2f} seconds\n{'='*40}")

if __name__ == "__main__":
    main()
