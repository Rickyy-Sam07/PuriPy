import re
import string
import logging
import pandas as pd
from typing import Optional, List, Tuple, Callable
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
import emoji
import contractions
from better_profanity import profanity
from textblob import TextBlob  
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
import numpy as np
from functools import partial
import sys
import time

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("textapp.log", mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logging.captureWarnings(True)

# ---------------------- NLTK Resource Download ----------------------
def ensure_nltk_resources():
    """Download required NLTK resources if missing."""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource, quiet=True)
            logger.info(f"Downloaded NLTK resource: {resource}")

ensure_nltk_resources()

# ---------------------- Config Validation ----------------------
def validate_config(config: dict) -> None:
    """Validate configuration dictionary for required keys and values."""
    required_keys = {
        'lowercase', 'remove_punctuation', 'remove_stopwords', 'remove_urls',
        'remove_html', 'remove_emojis', 'remove_numbers', 'expand_contractions',
        'spelling_correction', 'lemmatize', 'stem', 'tokenize', 'ngram_range',
        'profanity_filter', 'language', 'custom_stopwords', 'custom_profanity'
    }
    
    if missing := required_keys - config.keys():
        logger.error(f"Missing config keys: {missing}")
        raise KeyError(f"Missing config keys: {missing}")
    
    if config['tokenize'] not in (None, 'word', 'sentence'):
        logger.error("tokenize must be None, 'word', or 'sentence'")
        raise ValueError("tokenize must be None, 'word', or 'sentence'")
    
    if not isinstance(config['ngram_range'], tuple) or len(config['ngram_range']) != 2:
        logger.error("ngram_range must be a tuple of (min_n, max_n)")
        raise ValueError("ngram_range must be a tuple of (min_n, max_n)")
    
    if config['language'] not in stopwords.fileids():
        logger.error(f"Unsupported language: {config['language']}")
        raise ValueError(f"Unsupported language: {config['language']}. Available: {stopwords.fileids()}")
    

    if config['spelling_correction'] and config['language'] not in ['en', 'english']:
        warnings.warn("TextBlob spelling correction only supports English. Disabling spelling correction.")
        config['spelling_correction'] = False

# ---------------------- URL Removal ----------------------
def improved_url_removal(text: str) -> str:
    """Remove URLs and domain-like patterns"""  
    url_pattern = r'(https?://\S+|www\.\S+|(?:\b\w+\.)+\w{2,})'
    return re.sub(url_pattern, '', text, flags=re.IGNORECASE)

# ---------------------- Core Cleaning Function ----------------------
def clean(
    text: str,
    config: dict,
    stop_words: set,
    lemmatizer: Optional[WordNetLemmatizer],
    stemmer: Optional[PorterStemmer]
) -> str:  
    """Apply all configured cleaning steps to a single text string."""
    try:
        if pd.isna(text):
            return ""

        # Track original text for error reporting
        original_text = text[:100] + '...' if len(text) > 100 else text
        
        # 1. Expand contractions
        if config['expand_contractions']:
            text = contractions.fix(text)
            
        # 2. Remove URLs
        if config['remove_urls']:
            text = improved_url_removal(text)
            
        # 3. Remove HTML tags
        if config['remove_html']:
            text = re.sub(r'<.*?>', '', text)
            
        # 4. Remove emojis
        if config['remove_emojis']:
            text = emoji.replace_emoji(text, replace='')
            
        # 5. Remove numbers
        if config['remove_numbers']:
            text = re.sub(r'\d+', '', text)
            
        # 6. Lowercase
        if config['lowercase']:
            text = text.lower()
            
        # 7. Remove punctuation
        if config['remove_punctuation']:
            text = text.translate(str.maketrans('', '', string.punctuation))
            
        # 8. Tokenization & NLP processing
        if config['tokenize'] == 'word':
            tokens = word_tokenize(text)
            
            # Spelling correction with TextBlob
            if config['spelling_correction']:
                tokens = [str(TextBlob(w).correct()) if w.isalpha() else w for w in tokens]
                
            if config['remove_stopwords']:
                tokens = [w for w in tokens if w not in stop_words]
                
            if config['lemmatize']:
                tokens = [lemmatizer.lemmatize(w) for w in tokens]
                
            if config['stem']:
                tokens = [stemmer.stem(w) for w in tokens]
                
            if config['ngram_range'] != (1, 1):
                ngram_min, ngram_max = config['ngram_range']
                tokens = [
                    ' '.join(gram)
                    for n in range(ngram_min, ngram_max + 1)
                    for gram in ngrams(tokens, n)
                ]
                
            text = ' '.join(tokens)
            
        elif config['tokenize'] == 'sentence':
            sentences = sent_tokenize(text)
            text = ' '.join(sentences)
            
        else:
            if config['ngram_range'] != (1, 1):
                tokens = text.split()
                ngram_min, ngram_max = config['ngram_range']
                tokens = [
                    ' '.join(gram)
                    for n in range(ngram_min, ngram_max + 1)
                    for gram in ngrams(tokens, n)
                ]
                text = ' '.join(tokens)

        # 9. Profanity filter
        if config['profanity_filter']:
            text = profanity.censor(text)
            
        # 10. Final cleanup: normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    except Exception as e:
        logger.error(f"Error processing text: {str(e)} | Original: {original_text}")
        return text

# ---------------------- Cleaner Factory ----------------------
def clean_text_factory(config: dict) -> Callable[[str], str]:
    """Create a text cleaner with validated configuration."""
    validate_config(config)
    
    
    stop_words = set()
    if config['remove_stopwords'] or config['profanity_filter']:
        stop_words = set(stopwords.words(config['language']))
        if config.get('custom_stopwords'):
            stop_words.update(config['custom_stopwords'])
            
    lemmatizer = WordNetLemmatizer() if config['lemmatize'] else None
    stemmer = PorterStemmer() if config['stem'] else None
    
    if config['profanity_filter']:
        profanity.load_censor_words()
        if config.get('custom_profanity'):
            profanity.add_censor_words(config['custom_profanity'])
    
    return partial(clean, config=config, stop_words=stop_words,
                  lemmatizer=lemmatizer, stemmer=stemmer)

# ---------------------- DataFrame Cleaning ----------------------
def clean_dataframe(
    df: pd.DataFrame,
    cleaner: Callable[[str], str],
    text_column: Optional[str] = None,
    output_column: str = 'cleaned_text',
    parallel: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """Apply text cleaning to a DataFrame, optionally in parallel."""
    try:
        if text_column is None:
            text_column = auto_detect_text_column(df)
            logger.info(f"Auto-detected text column: {text_column}")
        if parallel:
            return _parallel_clean(df, cleaner, text_column, output_column, verbose)
        tqdm.pandas(desc="Cleaning text", disable=not verbose)
        df[output_column] = df[text_column].astype(str).progress_apply(cleaner)
        logger.info(f"Completed cleaning {len(df)} rows (sequential)")
        return df
    except Exception as e:
        logger.error(f"DataFrame cleaning failed: {str(e)}")
        raise

def _parallel_clean(
    df: pd.DataFrame,
    cleaner: Callable[[str], str],
    text_column: str,
    output_column: str,
    verbose: bool
) -> pd.DataFrame:
    """Parallel processing for DataFrame cleaning."""
    try:
        chunks = np.array_split(df, cpu_count())
        logger.info(f"Processing {len(chunks)} chunks across {cpu_count()} cores")
        with Pool() as pool:
            results = list(tqdm(
                pool.imap(_process_chunk, [
                    (chunk, cleaner, text_column, output_column)
                    for chunk in chunks
                ]),
                total=len(chunks),
                desc="Processing chunks",
                disable=not verbose
            ))
        logger.info("Completed parallel cleaning")
        return pd.concat(results)
    except Exception as e:
        logger.error(f"Parallel cleaning failed: {str(e)}")
        raise

def _process_chunk(args: Tuple[pd.DataFrame, Callable, str, str]) -> pd.DataFrame:
    """Helper for parallel chunk processing."""
    chunk, cleaner, text_col, out_col = args
    try:
        chunk[out_col] = chunk[text_col].astype(str).apply(cleaner)
        return chunk
    except Exception as e:
        logger.error(f"Chunk processing failed: {str(e)}")
        return chunk

# ---------------------- Auto Text Column Detection ----------------------
def auto_detect_text_column(df: pd.DataFrame) -> str:
    """Automatically detect the main text column in a DataFrame."""
    text_cols = df.select_dtypes(include=['object', 'string']).columns
    text_candidates = [col for col in text_cols if 'text' in col.lower()]
    if text_candidates:
        return text_candidates[0]
    if len(text_cols) == 1:
        return text_cols[0]
    word_counts = {col: df[col].astype(str).str.split().str.len().mean() for col in text_cols}
    return max(word_counts, key=word_counts.get)

# ---------------------- Report Generation ----------------------
def save_text_cleaning_report_as_txt(
    df: pd.DataFrame,
    text_column: str,
    output_column: str,
    config: dict = None,
    file_path: str = "text_cleaning_report.txt",
    include_samples: int = 5
) -> bool:
    """
    Generate and save a text cleaning report to a text file
    
    Args:
        df: DataFrame containing the original and cleaned text
        text_column: Name of the column containing the original text
        output_column: Name of the column containing the cleaned text
        config: Configuration dictionary used for cleaning (optional)
        file_path: Path where to save the text report
        include_samples: Number of text samples to include in report (0 to disable)
        
    Returns:
        bool: True if report was successfully saved, False otherwise
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("==================================================\n")
            f.write("             TEXT CLEANING REPORT                 \n")
            f.write("==================================================\n\n")
            
            # Dataset info
            f.write("DATASET INFORMATION:\n")
            f.write(f"Rows: {len(df)}\n")
            f.write(f"Original column: {text_column}\n")
            f.write(f"Cleaned column: {output_column}\n\n")
            
            # Configuration used
            if config:
                f.write("CLEANING CONFIGURATION:\n")
                for key, value in config.items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")
            
            # Statistics
            f.write("CLEANING STATISTICS:\n")
            
            # Calculate character and word count before and after
            char_count_before = df[text_column].astype(str).str.len().sum()
            char_count_after = df[output_column].astype(str).str.len().sum()
            
            word_count_before = df[text_column].astype(str).str.split().str.len().sum()
            word_count_after = df[output_column].astype(str).str.split().str.len().sum()
            
            char_reduction = 100 * (1 - char_count_after / char_count_before) if char_count_before > 0 else 0
            word_reduction = 100 * (1 - word_count_after / word_count_before) if word_count_before > 0 else 0
            
            f.write(f"Characters before cleaning: {char_count_before:,}\n")
            f.write(f"Characters after cleaning: {char_count_after:,}\n")
            f.write(f"Character reduction: {char_reduction:.2f}%\n\n")
            
            f.write(f"Words before cleaning: {word_count_before:,}\n")
            f.write(f"Words after cleaning: {word_count_after:,}\n")
            f.write(f"Word reduction: {word_reduction:.2f}%\n\n")
            
            # Empty values
            empty_before = df[df[text_column].isna() | (df[text_column].astype(str) == '')].shape[0]
            empty_after = df[df[output_column].isna() | (df[output_column].astype(str) == '')].shape[0]
            
            f.write(f"Empty values before: {empty_before} ({100 * empty_before / len(df):.2f}%)\n")
            f.write(f"Empty values after: {empty_after} ({100 * empty_after / len(df):.2f}%)\n\n")
            
            # Unique values
            unique_before = df[text_column].nunique()
            unique_after = df[output_column].nunique()
            
            f.write(f"Unique values before: {unique_before}\n")
            f.write(f"Unique values after: {unique_after}\n\n")
            
            # Average length
            avg_len_before = df[text_column].astype(str).str.len().mean()
            avg_len_after = df[output_column].astype(str).str.len().mean()
            
            f.write(f"Average length before: {avg_len_before:.2f} characters\n")
            f.write(f"Average length after: {avg_len_after:.2f} characters\n\n")
            
            # Sample before and after
            if include_samples > 0:
                f.write("SAMPLE TRANSFORMATIONS:\n")
                sample_size = min(include_samples, len(df))
                sample_df = df.sample(sample_size)
                
                for i in range(sample_size):
                    sample = sample_df.iloc[i]
                    f.write(f"Sample {i+1}:\n")
                    before = str(sample[text_column])
                    after = str(sample[output_column])
                    f.write(f"Before: {before[:100]}{'...' if len(before) > 100 else ''}\n")
                    f.write(f"After: {after[:100]}{'...' if len(after) > 100 else ''}\n\n")
            
            # Most common tokens (if feasible)
            try:
                from collections import Counter
                all_words = ' '.join(df[output_column].astype(str)).split()
                if len(all_words) > 0:
                    word_counts = Counter(all_words).most_common(10)
                    f.write("TOP 10 TOKENS AFTER CLEANING:\n")
                    for word, count in word_counts:
                        f.write(f"- {word}: {count} occurrences\n")
                    f.write("\n")
            except:
                pass
            
            f.write("==================================================\n")
            f.write("End of report\n")
            
        logger.info(f"Text cleaning report saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save text cleaning report: {str(e)}")
        return False


def clean_with_report(
    df: pd.DataFrame,
    config: dict,
    text_column: Optional[str] = None,
    output_column: str = 'cleaned_text',
    report_path: str = "text_cleaning_report.txt",
    parallel: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Clean text data and generate a comprehensive report
    
    Args:
        df: Input DataFrame
        config: Cleaning configuration dictionary
        text_column: Column containing text to clean (auto-detected if None)
        output_column: Column name for cleaned text
        report_path: Path to save the report (None to skip report)
        parallel: Whether to use parallel processing
        verbose: Whether to show progress bars
        
    Returns:
        DataFrame with cleaned text
    """
    start_time = time.time()
    
    try:
        # Auto-detect text column if not specified
        if text_column is None:
            text_column = auto_detect_text_column(df)
            logger.info(f"Auto-detected text column: {text_column}")
        
        # Create and apply the cleaner
        cleaner = create_text_cleaner(config)
        cleaned_df = clean_dataframe(
            df, 
            cleaner, 
            text_column=text_column,
            output_column=output_column,
            parallel=parallel, 
            verbose=verbose
        )
        
        # Generate report if path is provided
        if report_path:
            # Add execution time to config for reporting
            report_config = config.copy() if config else {}
            report_config['execution_time'] = f"{time.time() - start_time:.2f} seconds"
            report_config['parallel_processing'] = parallel
            report_config['cpu_cores'] = cpu_count() if parallel else 1
            
            save_text_cleaning_report_as_txt(
                cleaned_df,
                text_column,
                output_column,
                report_config,
                report_path
            )
            
        logger.info(f"Text cleaning completed in {time.time() - start_time:.2f} seconds")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Text cleaning failed: {str(e)}")
        raise

# ---------------------- Public API ----------------------
def create_text_cleaner(config: dict) -> Callable[[str], str]:
    """Public API to create a text cleaner."""
    return clean_text_factory(config)

