# dt.py - Automated date and time data cleaning utility
import pandas as pd
import numpy as np
import datetime as dt
import logging
import pytz
import warnings
from typing import Optional, List, Dict, Union, Tuple
from dateutil import parser
from pandas.api.types import is_datetime64_any_dtype
from statsmodels.tsa.seasonal import STL
from tqdm.auto import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('datetime_cleaner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DateTimeCleaner:
    """Automated date and time data cleaning with advanced features"""
    
    def __init__(self, df: pd.DataFrame, memory_efficient: bool = False):
        """
        Initialize the date/time cleaner
        
        Args:
            df: Input DataFrame
            memory_efficient: If True, operate in memory-efficient mode
        """
        try:
            self.df = df if memory_efficient else df.copy()
            self.memory_efficient = memory_efficient
            self.report = {}
            self.cleaned_columns = []
            self.fiscal_year_start = 10  # Default: October (month 10)
            
            logger.info(f"Initialized DateTimeCleaner with dataframe of shape {df.shape}")
        except Exception as e:
            logger.error(f"Failed to initialize date cleaner: {str(e)}")
            raise
    
    def auto_detect_datetime_columns(self, sample_size: int = 1000) -> List[str]:
        """
        Automatically detect columns that may contain date/time information
        
        Args:
            sample_size: Number of rows to sample for detection
            
        Returns:
            List of column names that likely contain date/time data
        """
        try:
            datetime_cols = []
            
            # Columns already in datetime format
            for col in self.df.columns:
                if is_datetime64_any_dtype(self.df[col]):
                    datetime_cols.append(col)
            
            # Sample the DataFrame for efficient processing
            sample_df = self.df.sample(min(sample_size, len(self.df))) if len(self.df) > sample_size else self.df
            
            # Check string columns that might contain dates
            for col in sample_df.columns:
                if col in datetime_cols:
                    continue
                    
                if self.df[col].dtype == 'object':
                    # Check if the first non-null value can be parsed as a date
                    sample_values = sample_df[col].dropna().astype(str).head(20).tolist()
                    
                    if not sample_values:
                        continue
                        
                    try:
                        # Test if at least 80% of the samples can be parsed as dates
                        success_count = 0
                        for val in sample_values:
                            if val and len(val) > 3:  # Avoid very short strings
                                try:
                                    parser.parse(val, fuzzy=True)
                                    success_count += 1
                                except:
                                    pass
                                    
                        if success_count >= len(sample_values) * 0.8:
                            datetime_cols.append(col)
                    except Exception:
                        pass
                        
                # Check numeric columns that might represent timestamps
                elif pd.api.types.is_numeric_dtype(self.df[col]):
                    # Check for Unix timestamps (seconds since epoch)
                    sample_values = sample_df[col].dropna().head(5).tolist()
                    
                    if not sample_values:
                        continue
                        
                    try:
                        # Check if the values are in a reasonable timestamp range
                        values_in_range = [
                            1000000000 < val < 2000000000 for val in sample_values
                        ]
                        if all(values_in_range):
                            datetime_cols.append(col)
                    except:
                        pass
            
            logger.info(f"Detected {len(datetime_cols)} potential datetime columns: {datetime_cols}")            
            return datetime_cols
            
        except Exception as e:
            logger.error(f"Error detecting datetime columns: {str(e)}")
            return []
    
    def parse_datetime(self, column: str, 
                      format_str: Optional[str] = None,
                      fuzzy: bool = True,
                      errors: str = 'coerce',
                      infer_format: bool = True) -> pd.DataFrame:
        """
        Parse column to datetime with flexible format detection
        
        Args:
            column: Column name to parse
            format_str: Optional explicit datetime format string
            fuzzy: Whether to use fuzzy parsing (ignore unknown parts)
            errors: How to handle parsing errors ('coerce', 'raise', or 'ignore')
            infer_format: Whether to try to infer format from the data
            
        Returns:
            DataFrame with parsed datetime column
        """
        try:
            if column not in self.df.columns:
                logger.warning(f"Column {column} not found in dataframe")
                return self.df
                
            # Initialize report for this column
            if column not in self.report:
                self.report[column] = {
                    'original_dtype': str(self.df[column].dtype),
                    'actions': []
                }
                
            # Skip if already datetime
            if is_datetime64_any_dtype(self.df[column]):
                logger.info(f"Column {column} is already in datetime format")
                self.cleaned_columns.append(column)
                self.report[column]['actions'].append("Column already in datetime format")
                return self.df
                
            # Try to parse with specified format
            if format_str:
                try:
                    self.df[column] = pd.to_datetime(self.df[column], format=format_str, errors=errors)
                    logger.info(f"Parsed {column} to datetime using format '{format_str}'")
                    self.report[column]['actions'].append(f"Parsed to datetime using format '{format_str}'")
                    self.cleaned_columns.append(column)
                    return self.df
                except Exception as e:
                    logger.warning(f"Failed to parse {column} with format '{format_str}': {str(e)}")
                    if errors == 'raise':
                        raise
            
            # Try to infer format from data
            if infer_format:
                try:
                    # Sample non-null values
                    sample = self.df[column].dropna().astype(str).sample(min(10, len(self.df))).tolist()
                    if sample:
                        # Try to infer format from first non-empty value
                        inferred_format = None
                        for val in sample:
                            try:
                                dt_val = parser.parse(val, fuzzy=fuzzy)
                                # Test some common formats
                                common_formats = [
                                    '%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', 
                                    '%Y-%m-%d %H:%M:%S', '%m-%d-%Y', '%d-%m-%Y',
                                    '%Y%m%d'
                                ]
                                for fmt in common_formats:
                                    try:
                                        if dt_val.strftime(fmt) == dt_val.strftime(fmt):
                                            inferred_format = fmt
                                            break
                                    except:
                                        pass
                                if inferred_format:
                                    break
                            except:
                                pass
                                
                        if inferred_format:
                            try:
                                self.df[column] = pd.to_datetime(self.df[column], format=inferred_format, errors=errors)
                                logger.info(f"Parsed {column} to datetime using inferred format '{inferred_format}'")
                                self.report[column]['actions'].append(f"Parsed to datetime using inferred format '{inferred_format}'")
                                self.cleaned_columns.append(column)
                                return self.df
                            except Exception as e:
                                logger.warning(f"Failed with inferred format '{inferred_format}': {str(e)}")
                except Exception as e:
                    logger.warning(f"Format inference failed for {column}: {str(e)}")
            
            # Try flexible parsing as last resort
            try:
                if self.df[column].dtype == 'object':
                    # Convert to string first to handle any non-string objects
                    self.df[column] = self.df[column].astype(str)
                    
                # For numeric columns that might be Unix timestamps
                elif pd.api.types.is_numeric_dtype(self.df[column]):
                    # Check if values are in reasonable Unix timestamp range
                    avg_value = self.df[column].mean()
                    if 1000000000 < avg_value < 2000000000:  # Between 2001 and 2033
                        self.df[column] = pd.to_datetime(self.df[column], unit='s', errors=errors)
                        logger.info(f"Parsed {column} as Unix timestamps (seconds)")
                        self.report[column]['actions'].append("Parsed as Unix timestamps (seconds)")
                        self.cleaned_columns.append(column)
                        return self.df
                
                # General purpose parsing
                self.df[column] = pd.to_datetime(self.df[column], errors=errors, infer_datetime_format=True)
                logger.info(f"Parsed {column} to datetime using general parser")
                self.report[column]['actions'].append("Parsed to datetime using general parser")
                self.cleaned_columns.append(column)
                
            except Exception as e:
                logger.error(f"Failed to parse {column} to datetime: {str(e)}")
                if errors == 'raise':
                    raise
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error parsing datetime for column {column}: {str(e)}")
            if errors == 'raise':
                raise
            return self.df
    
    def impute_missing_dates(self, column: str, 
                           method: str = 'linear',
                           seasonal_period: Optional[int] = None,
                           reference_col: Optional[str] = None) -> pd.DataFrame:
        """
        Fill missing datetime values using various methods
        
        Args:
            column: Column name to impute
            method: Imputation method ('linear', 'forward', 'backward', 'seasonal', 'mode')
            seasonal_period: Period for seasonal decomposition (e.g., 7 for weekly, 12 for monthly)
            reference_col: Optional reference column for conditional imputation
            
        Returns:
            DataFrame with imputed datetime values
        """
        try:
            if column not in self.df.columns:
                logger.warning(f"Column {column} not found")
                return self.df
                
            # Initialize report for this column
            if column not in self.report:
                self.report[column] = {
                    'original_dtype': str(self.df[column].dtype),
                    'actions': []
                }
                
            # Ensure column is datetime type
            if not is_datetime64_any_dtype(self.df[column]):
                logger.warning(f"Column {column} is not datetime type, parsing first")
                self.parse_datetime(column)
                
            # Skip if no missing values
            missing_count = self.df[column].isna().sum()
            if missing_count == 0:
                logger.info(f"No missing values in column {column}")
                self.report[column]['actions'].append("No missing values to impute")
                return self.df
                
            logger.info(f"Imputing {missing_count} missing values in {column} using {method}")
            
            # Apply the specified imputation method
            if method == 'mode':
                # Use most common date
                mode_value = self.df[column].mode()[0]
                self.df[column] = self.df[column].fillna(mode_value)
                self.report[column]['actions'].append(f"Imputed {missing_count} missing values with mode")
                
            elif method == 'forward':
                # Forward fill (use previous value)
                self.df[column] = self.df[column].fillna(method='ffill')
                self.report[column]['actions'].append(f"Imputed {missing_count} missing values with forward fill")
                
            elif method == 'backward':
                # Backward fill (use next value)
                self.df[column] = self.df[column].fillna(method='bfill')
                self.report[column]['actions'].append(f"Imputed {missing_count} missing values with backward fill")
                
            elif method == 'seasonal' and seasonal_period:
                # Use seasonal decomposition for imputation
                # This requires a time series index
                if reference_col and reference_col in self.df.columns:
                    # Use reference column for grouping
                    for group_val, group_df in self.df.groupby(reference_col):
                        if group_df[column].isna().any():
                            # Create time series
                            ts = group_df[column].copy()
                            # Apply seasonal decomposition and imputation
                            if len(ts.dropna()) >= 2 * seasonal_period:
                                try:
                                    # Interpolate missing values first to create a complete series
                                    ts_filled = ts.interpolate(method='linear')
                                    # Apply seasonal decomposition
                                    stl = STL(ts_filled, seasonal=seasonal_period)
                                    result = stl.fit()
                                    # Extract components
                                    trend = result.trend
                                    seasonal = result.seasonal
                                    # Recreate the series and fill missing values
                                    filled = trend + seasonal
                                    # Update original series
                                    self.df.loc[ts.index[ts.isna()], column] = filled[ts.isna()]
                                except Exception as e:
                                    logger.warning(f"Seasonal decomposition failed: {str(e)}")
                                    # Fall back to linear interpolation
                                    self.df.loc[group_df.index, column] = group_df[column].interpolate(method='linear')
                            else:
                                # Not enough data for seasonal decomposition
                                self.df.loc[group_df.index, column] = group_df[column].interpolate(method='linear')
                else:
                    # Without reference column, treat whole dataset as one time series
                    if len(self.df[column].dropna()) >= 2 * seasonal_period:
                        try:
                            # Create a temporary datetime index if needed
                            if not isinstance(self.df.index, pd.DatetimeIndex):
                                temp_df = self.df.copy()
                                temp_df.set_index(column, inplace=True)
                                temp_df = temp_df.sort_index()
                                # Forward fill to have a complete index
                                temp_series = temp_df.index.to_series()
                                temp_series = temp_series.interpolate(method='linear')
                                # Apply STL decomposition
                                stl = STL(temp_series, seasonal=seasonal_period)
                                result = stl.fit()
                                # Extract components
                                trend = result.trend
                                seasonal = result.seasonal
                                # Recreate the series
                                filled = trend + seasonal
                                # Map back to original dataframe
                                # This is complex and may need custom logic depending on the data
                                logger.warning("Seasonal imputation without reference column may not be accurate")
                                # Fall back to linear interpolation
                                self.df[column] = self.df[column].interpolate(method='linear')
                            else:
                                # The DataFrame already has a datetime index
                                ts = self.df[column].copy()
                                ts_filled = ts.interpolate(method='linear')
                                stl = STL(ts_filled, seasonal=seasonal_period)
                                result = stl.fit()
                                trend = result.trend
                                seasonal = result.seasonal
                                filled = trend + seasonal
                                self.df.loc[ts.isna(), column] = filled[ts.isna()]
                        except Exception as e:
                            logger.warning(f"Seasonal decomposition failed: {str(e)}")
                            # Fall back to linear interpolation
                            self.df[column] = self.df[column].interpolate(method='linear')
                    else:
                        # Not enough data for seasonal decomposition
                        self.df[column] = self.df[column].interpolate(method='linear')
                        
                self.report[column]['actions'].append(f"Imputed {missing_count} missing values with seasonal decomposition (period={seasonal_period})")
                
            else:
                # Default to linear interpolation
                self.df[column] = self.df[column].interpolate(method='linear')
                self.report[column]['actions'].append(f"Imputed {missing_count} missing values with linear interpolation")
            
            # Handle any remaining NaN values at the edges
            na_count = self.df[column].isna().sum()
            if na_count > 0:
                if na_count == self.df[column].isna().sum():
                    logger.warning("Imputation didn't fill any values, using forward/backward fill")
                    self.df[column] = self.df[column].fillna(method='ffill').fillna(method='bfill')
                else:
                    # Fill any remaining NA values
                    self.df[column] = self.df[column].fillna(method='ffill').fillna(method='bfill')
                    
            logger.info(f"Completed imputation for {column}")
            return self.df
            
        except Exception as e:
            logger.error(f"Error imputing missing dates for {column}: {str(e)}")
            return self.df
    
    def standardize_timezone(self, column: str, 
                           target_timezone: str = 'UTC',
                           source_timezone: Optional[str] = None,
                           ambiguous: str = 'raise') -> pd.DataFrame:
        """
        Convert datetime column to a target timezone
        
        Args:
            column: Column name to convert
            target_timezone: Target timezone (e.g. 'UTC', 'US/Eastern')
            source_timezone: Source timezone if data doesn't include it
            ambiguous: How to handle ambiguous times
            
        Returns:
            DataFrame with timezone-standardized column
        """
        try:
            if column not in self.df.columns:
                logger.warning(f"Column {column} not found")
                return self.df
                
            # Initialize report for this column
            if column not in self.report:
                self.report[column] = {
                    'original_dtype': str(self.df[column].dtype),
                    'actions': []
                }
                
            # Ensure the column is datetime type
            if not is_datetime64_any_dtype(self.df[column]):
                logger.warning(f"Column {column} is not datetime type, parsing first")
                self.parse_datetime(column)
                
            # Check if the datetimes already have timezone info
            has_tz = False
            if hasattr(self.df[column].dtype, 'tz'):
                has_tz = self.df[column].dtype.tz is not None
                
            # Get valid timezone
            try:
                target_tz = pytz.timezone(target_timezone)
            except:
                logger.warning(f"Invalid target timezone: {target_timezone}, using UTC")
                target_tz = pytz.UTC
                target_timezone = 'UTC'
            
            # Handle timezone conversion
            if has_tz:
                # If data already has timezone info, convert directly
                self.df[column] = self.df[column].dt.tz_convert(target_timezone)
                current_tz = self.df[column].dt.tz
                logger.info(f"Converted {column} from {current_tz} to {target_timezone}")
                self.report[column]['actions'].append(f"Converted timezone from {current_tz} to {target_timezone}")
            elif source_timezone:
                # If data doesn't have timezone but source is specified
                try:
                    source_tz = pytz.timezone(source_timezone)
                    self.df[column] = self.df[column].dt.tz_localize(
                        source_timezone, ambiguous=ambiguous
                    ).dt.tz_convert(target_timezone)
                    logger.info(f"Localized {column} from {source_timezone} to {target_timezone}")
                    self.report[column]['actions'].append(f"Localized from {source_timezone} to {target_timezone}")
                except Exception as e:
                    logger.error(f"Timezone conversion error: {str(e)}")
                    # Try with ambiguous='NaT' as a fallback
                    try:
                        self.df[column] = self.df[column].dt.tz_localize(
                            source_timezone, ambiguous='NaT'
                        ).dt.tz_convert(target_timezone)
                        logger.info(f"Localized {column} with NaT handling from {source_timezone} to {target_timezone}")
                        self.report[column]['actions'].append(
                            f"Localized with NaT handling from {source_timezone} to {target_timezone}"
                        )
                    except Exception as e2:
                        logger.error(f"Fallback timezone conversion failed: {str(e2)}")
            else:
                # If no source timezone, assume UTC
                try:
                    self.df[column] = self.df[column].dt.tz_localize('UTC').dt.tz_convert(target_timezone)
                    logger.info(f"Assumed UTC and converted {column} to {target_timezone}")
                    self.report[column]['actions'].append(f"Assumed UTC and converted to {target_timezone}")
                except Exception as e:
                    logger.error(f"Timezone conversion error: {str(e)}")
                    
            self.cleaned_columns.append(column)
            return self.df
            
        except Exception as e:
            logger.error(f"Error standardizing timezone for {column}: {str(e)}")
            return self.df
    
    def extract_calendar_features(self, column: str, 
                                features: Optional[List[str]] = None,
                                fiscal_year_start: Optional[int] = None) -> pd.DataFrame:
        """
        Extract calendar features from datetime column
        
        Args:
            column: Datetime column name
            features: List of features to extract, or None for all
            fiscal_year_start: Starting month for fiscal year (1-12)
            
        Returns:
            DataFrame with added calendar features
        """
        try:
            if column not in self.df.columns:
                logger.warning(f"Column {column} not found")
                return self.df
                
            # Initialize report for this column
            if column not in self.report:
                self.report[column] = {
                    'original_dtype': str(self.df[column].dtype),
                    'actions': []
                }
                
            # Ensure column is datetime type
            if not is_datetime64_any_dtype(self.df[column]):
                logger.warning(f"Column {column} is not datetime type, parsing first")
                self.parse_datetime(column)
            
            # Set default features if none provided
            if features is None:
                features = [
                    'year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend',
                    'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end',
                    'is_year_start', 'is_year_end', 'hour', 'minute'
                ]
                
            # Keep track of created features
            created_features = []
            
            # Set fiscal year start if provided
            if fiscal_year_start is not None:
                if 1 <= fiscal_year_start <= 12:
                    self.fiscal_year_start = fiscal_year_start
                else:
                    logger.warning(f"Invalid fiscal_year_start {fiscal_year_start}, using default ({self.fiscal_year_start})")
                    
            # Extract basic calendar components
            base_features = {
                'year': ('dt.year', lambda x: x.dt.year),
                'month': ('dt.month', lambda x: x.dt.month),
                'day': ('dt.day', lambda x: x.dt.day),
                'dayofweek': ('dt.dayofweek', lambda x: x.dt.dayofweek),
                'quarter': ('dt.quarter', lambda x: x.dt.quarter),
                'dayofyear': ('dt.dayofyear', lambda x: x.dt.dayofyear),
                'weekofyear': ('dt.isocalendar().week', lambda x: x.dt.isocalendar().week),
                'hour': ('dt.hour', lambda x: x.dt.hour),
                'minute': ('dt.minute', lambda x: x.dt.minute),
                'second': ('dt.second', lambda x: x.dt.second),
            }
            
            # Extract boolean indicators
            indicator_features = {
                'is_weekend': ('dt.dayofweek >= 5', lambda x: x.dt.dayofweek >= 5),
                'is_month_start': ('dt.is_month_start', lambda x: x.dt.is_month_start),
                'is_month_end': ('dt.is_month_end', lambda x: x.dt.is_month_end),
                'is_quarter_start': ('dt.is_quarter_start', lambda x: x.dt.is_quarter_start),
                'is_quarter_end': ('dt.is_quarter_end', lambda x: x.dt.is_quarter_end),
                'is_year_start': ('dt.is_year_start', lambda x: x.dt.is_year_start),
                'is_year_end': ('dt.is_year_end', lambda x: x.dt.is_year_end),
            }
            
            # Apply selected basic features
            for feature_name, (desc, func) in base_features.items():
                if feature_name in features:
                    try:
                        new_col = f"{column}_{feature_name}"
                        self.df[new_col] = func(self.df[column])
                        created_features.append(new_col)
                        logger.debug(f"Added {feature_name} feature to {column}")
                    except Exception as e:
                        logger.warning(f"Failed to create {feature_name} feature: {str(e)}")
            
            # Apply selected indicator features
            for feature_name, (desc, func) in indicator_features.items():
                if feature_name in features:
                    try:
                        new_col = f"{column}_{feature_name}"
                        self.df[new_col] = func(self.df[column]).astype(int)
                        created_features.append(new_col)
                        logger.debug(f"Added {feature_name} indicator to {column}")
                    except Exception as e:
                        logger.warning(f"Failed to create {feature_name} indicator: {str(e)}")
            
            # Custom features
            if 'fiscal_quarter' in features:
                try:
                    # Calculate fiscal quarter based on fiscal year start
                    fiscal_month = ((self.df[column].dt.month - self.fiscal_year_start) % 12) + 1
                    fiscal_quarter = ((fiscal_month - 1) // 3) + 1
                    self.df[f"{column}_fiscal_quarter"] = fiscal_quarter
                    created_features.append(f"{column}_fiscal_quarter")
                except Exception as e:
                    logger.warning(f"Failed to create fiscal_quarter: {str(e)}")
                    
            if 'fiscal_year' in features:
                try:
                    # Calculate fiscal year based on fiscal year start
                    cal_year = self.df[column].dt.year
                    month = self.df[column].dt.month
                    self.df[f"{column}_fiscal_year"] = cal_year.where(month >= self.fiscal_year_start, cal_year - 1)
                    created_features.append(f"{column}_fiscal_year")
                except Exception as e:
                    logger.warning(f"Failed to create fiscal_year: {str(e)}")
                    
            if 'season' in features:
                try:
                    # Define seasons: 1=Spring, 2=Summer, 3=Fall, 4=Winter
                    # Northern hemisphere definition
                    month_to_season = {
                        1: 4, 2: 4, 3: 1, 4: 1, 5: 1, 6: 2,
                        7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 4
                    }
                    self.df[f"{column}_season"] = self.df[column].dt.month.map(month_to_season)
                    created_features.append(f"{column}_season")
                except Exception as e:
                    logger.warning(f"Failed to create season: {str(e)}")
                    
            if 'business_day' in features:
                try:
                    # Check if date is a business day (Mon-Fri, not considering holidays)
                    self.df[f"{column}_business_day"] = (self.df[column].dt.dayofweek < 5).astype(int)
                    created_features.append(f"{column}_business_day")
                except Exception as e:
                    logger.warning(f"Failed to create business_day: {str(e)}")
                    
            if created_features:
                logger.info(f"Created {len(created_features)} calendar features for {column}: {created_features}")
                self.report[column]['actions'].append(f"Created {len(created_features)} calendar features")
                self.report[column]['created_features'] = created_features
                
            return self.df
            
        except Exception as e:
            logger.error(f"Error extracting calendar features from {column}: {str(e)}")
            return self.df
    
    def validate_temporal_consistency(self, start_column: str, 
                                    end_column: str,
                                    action: str = 'flag') -> pd.DataFrame:
        """
        Check and optionally fix temporal consistency (start <= end)
        
        Args:
            start_column: Name of the start date column
            end_column: Name of the end date column
            action: What to do with inconsistencies ('flag', 'swap', 'drop', 'truncate')
            
        Returns:
            DataFrame with consistency validation applied
        """
        try:
            if start_column not in self.df.columns or end_column not in self.df.columns:
                logger.warning(f"One or both columns not found: {start_column}, {end_column}")
                return self.df
                
            # Ensure both columns are datetime
            for col in [start_column, end_column]:
                if not is_datetime64_any_dtype(self.df[col]):
                    logger.info(f"Converting {col} to datetime")
                    self.parse_datetime(col)
            
            # Find violations where start_date > end_date
            mask = self.df[start_column] > self.df[end_column]
            violation_count = mask.sum()
            
            if violation_count == 0:
                logger.info(f"No temporal consistency violations between {start_column} and {end_column}")
                self.report[start_column] = self.report.get(start_column, {})
                self.report[end_column] = self.report.get(end_column, {})
                self.report[start_column]['actions'] = self.report[start_column].get('actions', [])
                self.report[end_column]['actions'] = self.report[end_column].get('actions', [])
                self.report[start_column]['actions'].append(f"No temporal consistency violations with {end_column}")
                self.report[end_column]['actions'].append(f"No temporal consistency violations with {start_column}")
                return self.df
                
            logger.warning(f"Found {violation_count} temporal consistency violations where {start_column} > {end_column}")
            
            # Add consistency flag column
            flag_col = f"{start_column}_{end_column}_consistent"
            self.df[flag_col] = (~mask).astype(int)
            
            # Apply the specified action
            if action == 'swap':
                # Swap the start and end dates where start > end
                temp = self.df.loc[mask, start_column].copy()
                self.df.loc[mask, start_column] = self.df.loc[mask, end_column]
                self.df.loc[mask, end_column] = temp
                logger.info(f"Swapped values for {violation_count} violations")
                action_desc = f"Swapped values for {violation_count} violations"
                
            elif action == 'drop':
                # Drop rows with inconsistencies
                self.df = self.df[~mask].copy()
                logger.info(f"Dropped {violation_count} rows with inconsistent dates")
                action_desc = f"Dropped {violation_count} rows with inconsistencies"
                
            elif action == 'truncate':
                # Truncate: set end_date = start_date where start > end
                self.df.loc[mask, end_column] = self.df.loc[mask, start_column]
                logger.info(f"Truncated {violation_count} inconsistent date ranges")
                action_desc = f"Truncated {violation_count} inconsistent ranges by setting end = start"
                
            else:  # 'flag' or any other value
                # Just keep the flag column (already created above)
                logger.info(f"Flagged {violation_count} temporal inconsistencies")
                action_desc = f"Flagged {violation_count} temporal inconsistencies"
            
            # Update report
            self.report[start_column] = self.report.get(start_column, {})
            self.report[end_column] = self.report.get(end_column, {})
            self.report[start_column]['actions'] = self.report[start_column].get('actions', [])
            self.report[end_column]['actions'] = self.report[end_column].get('actions', [])
            self.report[start_column]['actions'].append(action_desc)
            self.report[end_column]['actions'].append(action_desc)
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error validating temporal consistency: {str(e)}")
            return self.df
            
    def auto_clean_datetime_column(self, column: str,
                                 parse_dates: bool = True,
                                 impute_missing: bool = True,
                                 imputation_method: str = 'linear',
                                 standardize_tz: bool = False,
                                 target_tz: str = 'UTC',
                                 extract_features: bool = True,
                                 feature_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Complete datetime column cleaning pipeline
        
        Args:
            column: Column name to clean
            parse_dates: Whether to parse string dates to datetime
            impute_missing: Whether to impute missing values
            imputation_method: Method for imputation
            standardize_tz: Whether to standardize timezone
            target_tz: Target timezone for standardization
            extract_features: Whether to extract calendar features
            feature_list: List of calendar features to extract
            
        Returns:
            DataFrame with cleaned datetime column
        """
        try:
            logger.info(f"Starting auto-cleaning pipeline for datetime column: {column}")
            
            # Parse dates if needed
            if parse_dates:
                self.parse_datetime(column)
                
            # Impute missing values if needed
            if impute_missing:
                self.impute_missing_dates(column, method=imputation_method)
                
            # Standardize timezone if needed
            if standardize_tz:
                self.standardize_timezone(column, target_timezone=target_tz)
                
            # Extract calendar features if needed
            if extract_features:
                self.extract_calendar_features(column, features=feature_list)
                
            logger.info(f"Completed auto-cleaning pipeline for {column}")
            return self.df
            
        except Exception as e:
            logger.error(f"Error in auto_clean_datetime_column for {column}: {str(e)}")
            return self.df
    
    def get_report(self) -> Dict:
        """Generate a report of all cleaning actions performed"""
        return self.report


def clean_datetime_columns(df: pd.DataFrame, 
                         columns: Optional[List[str]] = None,
                         auto_detect: bool = True,
                         parse_dates: bool = True,
                         impute_missing: bool = True,
                         standardize_tz: bool = False,
                         extract_features: bool = True,
                         start_end_pairs: Optional[List[Tuple[str, str]]] = None,
                         **kwargs) -> pd.DataFrame:
    """
    Clean all datetime columns in a dataframe
    
    Args:
        df: Input dataframe
        columns: List of datetime columns to process (None to auto-detect)
        auto_detect: Whether to auto-detect datetime columns
        parse_dates: Whether to parse columns to datetime
        impute_missing: Whether to impute missing values
        standardize_tz: Whether to standardize timezones
        extract_features: Whether to extract calendar features
        start_end_pairs: List of (start, end) column tuples to validate consistency
        **kwargs: Additional arguments for DateTimeCleaner
        
    Returns:
        DataFrame with cleaned datetime columns
    """
    try:
        start_time = time.time()
        logger.info(f"Starting datetime cleaning process")
        
        # Initialize cleaner
        cleaner = DateTimeCleaner(df, memory_efficient=kwargs.get('memory_efficient', False))
        
        # Auto-detect datetime columns if needed
        datetime_columns = []
        if columns:
            datetime_columns = [col for col in columns if col in df.columns]
        elif auto_detect:
            datetime_columns = cleaner.auto_detect_datetime_columns()
            
        if not datetime_columns:
            logger.warning("No datetime columns found to process")
            return df
            
        logger.info(f"Processing {len(datetime_columns)} datetime columns: {datetime_columns}")
        
        # Process each column
        for col in tqdm(datetime_columns, desc="Cleaning datetime columns"):
            cleaner.auto_clean_datetime_column(
                col,
                parse_dates=parse_dates,
                impute_missing=impute_missing,
                imputation_method=kwargs.get('imputation_method', 'linear'),
                standardize_tz=standardize_tz,
                target_tz=kwargs.get('target_tz', 'UTC'),
                extract_features=extract_features,
                feature_list=kwargs.get('feature_list', None)
            )
            
        # Validate temporal consistency for start-end pairs
        if start_end_pairs:
            for start_col, end_col in start_end_pairs:
                if start_col in df.columns and end_col in df.columns:
                    cleaner.validate_temporal_consistency(
                        start_col, end_col, 
                        action=kwargs.get('consistency_action', 'flag')
                    )
        
        logger.info(f"Datetime cleaning completed in {time.time() - start_time:.2f} seconds")
        return cleaner.df
        
    except Exception as e:
        logger.error(f"Error in datetime cleaning process: {str(e)}")
        return df


def detect_date_format(date_str: str) -> str:
    """
    Detect the format of a date string
    
    Args:
        date_str: Date string to analyze
        
    Returns:
        Inferred date format string, or empty string if detection fails
    """
    try:
        # Common date formats to check
        formats = [
            '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y',
            '%m-%d-%Y', '%m/%d/%Y', '%Y%m%d',
            '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
            '%d-%m-%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S',
            '%m-%d-%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f'
        ]
        
        # Try to parse the date string with each format
        for fmt in formats:
            try:
                dt.datetime.strptime(date_str, fmt)
                return fmt
            except ValueError:
                continue
                
        # If no format worked, try using dateutil
        try:
            parsed = parser.parse(date_str)
            # If we got here, parsing worked, but we don't know the exact format
            # Try to guess based on the date components
            if 'T' in date_str:
                return '%Y-%m-%dT%H:%M:%S'
            elif '/' in date_str:
                if date_str.index('/') == 4:  # Year first
                    return '%Y/%m/%d %H:%M:%S' if ':' in date_str else '%Y/%m/%d'
                else:  # Day or month first
                    return '%d/%m/%Y %H:%M:%S' if ':' in date_str else '%d/%m/%Y'
            elif '-' in date_str:
                if date_str.index('-') == 4:  # Year first
                    return '%Y-%m-%d %H:%M:%S' if ':' in date_str else '%Y-%m-%d'
                else:  # Day or month first
                    return '%d-%m-%Y %H:%M:%S' if ':' in date_str else '%d-%m-%Y'
            elif len(date_str) == 8 and date_str.isdigit():
                return '%Y%m%d'
            else:
                return ''
        except:
            return ''
    except:
        return ''

def save_datetime_cleaning_report_as_text(cleaner: DateTimeCleaner, 
                                         file_path: str = "datetime_cleaning_report.txt",
                                         include_samples: int = 5) -> bool:
    """
    Save date/time cleaning report as a text file
    
    Args:
        cleaner: DateTimeCleaner instance with completed operations
        file_path: Path where to save the text report
        include_samples: Number of sample rows to include in the report
        
    Returns:
        Boolean indicating success
    """
    try:
        report = cleaner.get_report()
        
        with open(file_path, 'w') as f:
            # Header
            f.write("==================================================\n")
            f.write("          DATE & TIME CLEANING REPORT             \n")
            f.write("==================================================\n\n")
            
            # Dataset info
            f.write("DATASET INFORMATION:\n")
            f.write(f"Shape: {cleaner.df.shape[0]} rows x {cleaner.df.shape[1]} columns\n\n")
            
            # Date/time columns summary
            f.write("==================================================\n")
            f.write("               COLUMN SUMMARIES                   \n")
            f.write("==================================================\n\n")
            
            for col, details in report.items():
                f.write(f"COLUMN: {col}\n")
                f.write("-" * 50 + "\n")
                
                # Original data type
                f.write(f"Original type: {details.get('original_dtype', 'Unknown')}\n")
                
                # List of actions performed
                f.write("\nActions performed:\n")
                for i, action in enumerate(details.get('actions', []), 1):
                    f.write(f"  {i}. {action}\n")
                
                # List created features if any
                created_features = details.get('created_features', [])
                if created_features:
                    f.write(f"\nFeatures created ({len(created_features)}):\n")
                    for feature in created_features:
                        f.write(f"  - {feature}\n")
                        
                f.write("\n" + "=" * 50 + "\n\n")
                
            # Sample cleaned data
            if include_samples > 0:
                f.write("\n==================================================\n")
                f.write("               SAMPLE CLEANED DATA                \n")
                f.write("==================================================\n\n")
                
                # Get datetime columns that were cleaned
                datetime_cols = list(report.keys())
                
                # Select a subset of columns if there are many
                if len(datetime_cols) > 5:
                    display_cols = datetime_cols[:5]
                    f.write(f"Showing first 5 date/time columns (out of {len(datetime_cols)} total)\n\n")
                else:
                    display_cols = datetime_cols
                
                # Create a sample representation of data
                sample_data = cleaner.df[display_cols].head(include_samples).to_string()
                f.write(sample_data + "\n\n")
                
                # Note about generated features
                all_features = []
                for col, details in report.items():
                    all_features.extend(details.get('created_features', []))
                    
                if all_features:
                    f.write(f"\nTotal generated features: {len(all_features)}\n")
                    if len(all_features) <= 10:
                        f.write(f"Feature columns: {', '.join(all_features)}\n")
                    else:
                        f.write(f"First 10 feature columns: {', '.join(all_features[:10])}...\n")
                        
            f.write("\nEnd of report\n")
            
        logger.info(f"Date/time cleaning report saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save date/time cleaning report: {str(e)}")
        return False