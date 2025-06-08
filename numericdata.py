import pandas as pd
import numpy as np
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Callable, Optional
from functools import partial, reduce

# Enhanced logging configuration with rotation
def setup_logger():
    """Configure a logger with file rotation and proper formatting."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(log_dir, 'app.log')
    
    # Create rotating file handler (10MB max, keep 5 backup files)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logger()

def create_cleaning_pipeline(config: Dict) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Create a complete data cleaning pipeline from the provided configuration.
    
    This function constructs a sequence of data cleaning operations that will be
    applied in order to the input DataFrame. Each step is configured according
    to the settings in the config dictionary.
    
    Parameters:
        config (Dict): Configuration dictionary with settings for each cleaning step
    
    Returns:
        Callable[[pd.DataFrame], pd.DataFrame]: A function that accepts and returns a DataFrame
    
    Example:
        >>> pipeline = create_cleaning_pipeline(DEFAULT_CONFIG)
        >>> cleaned_df = pipeline(df)
    """
    try:
        # Create a sequence of data cleaning steps
        steps = [
            # Log initial state
            partial(log_cleaning_step, step_name="1. Initial Data"),
            
            # Step 1: Convert columns to proper numeric types
            partial(convert_data_types, **config.get('type_conversion', {})),
            partial(log_cleaning_step, step_name="2. After Type Conversion"),
            
            # Step 2: Apply business rules and fix constraint violations
            partial(correct_data_errors, **config.get('data_errors', {})),
            partial(log_cleaning_step, step_name="3. After Error Correction"),
            
            # Step 3: Handle missing values using the specified strategy
            partial(handle_missing_values, **config.get('missing_values', {})),
            partial(log_cleaning_step, step_name="4. After Missing Values"),
            
            # Step 4: Detect and handle outliers
            partial(detect_and_handle_outliers, **config.get('outliers', {})),
            partial(log_cleaning_step, step_name="5. After Outliers"),
            
            # Step 5: Handle duplicate rows
            partial(handle_duplicates, **config.get('duplicates', {})),
            partial(log_cleaning_step, step_name="6. After Duplicates"),
            
            # Step 6: Adjust numeric precision for each column
            # Note: Pass precision as a named parameter, not as **kwargs
            partial(adjust_precision, precision=config.get('precision', {})),
            partial(log_cleaning_step, step_name="7. Final Data")
        ]
        
        # Compose all steps into a single function
        logger.info("Created data cleaning pipeline with %d steps", len(steps))
        return compose(*steps)
        
    except Exception as e:
        logger.error(f"Failed to create cleaning pipeline: {str(e)}", exc_info=True)
        raise

def compose(*functions: Callable) -> Callable:
    """Compose multiple functions into a single function."""
    return reduce(lambda f, g: lambda x: g(f(x)), functions)

def log_cleaning_step(df: pd.DataFrame, step_name: str) -> pd.DataFrame:
    """Log the current state of the DataFrame."""
    logger.info(f"{step_name}: Shape {df.shape}")
    # Log sample of first numeric columns for debugging
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]
    for col in numeric_cols:
        if col in df.columns:
            logger.info(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    return df

def convert_data_types(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
    """
    Convert specified columns to numeric data types, handling non-numeric values.
    
    This function attempts to convert the specified columns to numeric types,
    replacing any non-convertible values with NaN.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        numeric_cols (List[str], optional): List of column names to convert
        **kwargs: Additional arguments (not used)
        
    Returns:
        pd.DataFrame: DataFrame with converted columns
        
    Example:
        >>> df = convert_data_types(df, numeric_cols=['age', 'salary'])
    """
    try:
        if numeric_cols is None:
            logger.info("No columns specified for type conversion")
            return df
            
        for col in numeric_cols:
            if col in df.columns:
                # Track conversion results
                original_na_count = df[col].isna().sum()
                df[col] = pd.to_numeric(df[col], errors='coerce')
                new_na_count = df[col].isna().sum()
                converted_to_na = new_na_count - original_na_count
                
                logger.info(f"Converted {col} to numeric ({converted_to_na} values became NaN)")
        
        return df
        
    except Exception as e:
        logger.error(f"Type conversion failed: {str(e)}", exc_info=True)
        return df

def correct_data_errors(df: pd.DataFrame, constraints: Optional[Dict[str, Callable]] = None, 
                       correction: str = 'median', **kwargs) -> pd.DataFrame:
    """Correct data that violates business rules BEFORE handling missing values."""
    try:
        if constraints is None:
            return df
            
        for col, constraint_func in constraints.items():
            if col not in df.columns:
                continue
                
            # Apply constraints only to non-NaN values
            non_na_mask = df[col].notna()
            if non_na_mask.sum() == 0:
                continue
                
            # Check constraints on non-NaN values
            valid_mask = non_na_mask & constraint_func(df[col])
            invalid_mask = non_na_mask & ~constraint_func(df[col])
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                logger.info(f"Found {invalid_count} constraint violations in {col}")
                
                # Calculate replacement from valid non-NaN data
                valid_data = df.loc[valid_mask, col]
                if len(valid_data) > 0:
                    if correction == 'median':
                        replacement = valid_data.median()
                    elif correction == 'mean':
                        replacement = valid_data.mean()
                    else:
                        replacement = valid_data.mode().iloc[0] if not valid_data.mode().empty else valid_data.mean()
                    
                    # Apply correction only to invalid values (not NaN)
                    df.loc[invalid_mask, col] = replacement
                    logger.info(f"Corrected {invalid_count} invalid values in {col} with {replacement:.2f}")
        
        return df
        
    except Exception as e:
        logger.error(f"Data error correction failed: {str(e)}")
        return df

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean', threshold: float = 0.5, **kwargs) -> pd.DataFrame:
    """Handle missing values using specified strategy."""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                logger.info(f"Handling {missing_count} missing values in {col}")
                
                # Calculate fill value from existing data
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    if strategy == 'mean':
                        fill_value = valid_data.mean()
                    elif strategy == 'median':
                        fill_value = valid_data.median()
                    else:
                        fill_value = valid_data.mode().iloc[0] if not valid_data.mode().empty else valid_data.mean()
                    
                    df[col] = df[col].fillna(fill_value)
                    logger.info(f"Filled missing values in {col} with {fill_value:.2f}")
        
        return df
        
    except Exception as e:
        logger.error(f"Missing value handling failed: {str(e)}")
        return df

def detect_and_handle_outliers(df: pd.DataFrame, method: str = 'iqr', action: str = 'cap', 
                              columns: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
    """Detect and handle outliers using IQR method."""
    try:
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0:
                    logger.info(f"Found {outliers_count} outliers in {col} (bounds: {lower_bound:.0f} - {upper_bound:.0f})")
                    
                    if action == 'cap':
                        df.loc[df[col] < lower_bound, col] = lower_bound
                        df.loc[df[col] > upper_bound, col] = upper_bound
                        logger.info(f"Capped outliers in {col}")
        
        return df
        
    except Exception as e:
        logger.error(f"Outlier handling failed: {str(e)}")
        return df

def handle_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = 'first', **kwargs) -> pd.DataFrame:
    """Remove duplicate rows."""
    try:
        initial_count = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate rows")
        
        return df
        
    except Exception as e:
        logger.error(f"Duplicate handling failed: {str(e)}")
        return df

def adjust_precision(df: pd.DataFrame, precision: Optional[Dict[str, int]] = None, **kwargs) -> pd.DataFrame:
    """
    Adjust numerical precision for specified columns.
    
    This function rounds numeric columns to the specified number of decimal places
    and converts columns with 0 decimal places to integers.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        precision (Dict[str, int], optional): Dictionary mapping column names to decimal places
        **kwargs: Additional arguments (not used)
        
    Returns:
        pd.DataFrame: DataFrame with adjusted precision
        
    Example:
        >>> precision_config = {'age': 0, 'salary': 0, 'score': 1}
        >>> df = adjust_precision(df, precision=precision_config)
    """
    try:
        if precision is None or not precision:
            logger.info("No precision configuration provided")
            return df
            
        logger.info(f"Adjusting precision for {len(precision)} columns: {list(precision.keys())}")
        
        for col, decimals in precision.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Round first to specified decimal places
                df[col] = df[col].round(decimals)
                
                if decimals == 0:
                    # Convert to integer for 0 decimal places
                    df[col] = df[col].astype(int)
                    logger.info(f"Adjusted precision of {col} to integer (0 decimals)")
                else:
                    logger.info(f"Adjusted precision of {col} to {decimals} decimal places")
            else:
                logger.warning(f"Column {col} not found or not numeric - cannot adjust precision")
        
        return df
        
    except Exception as e:
        logger.error(f"Precision adjustment failed: {str(e)}", exc_info=True)
        return df


