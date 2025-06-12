import pandas as pd
import numpy as np
import logging
import os
import warnings
import time
from functools import lru_cache
from joblib import Parallel, delayed
import gc
from typing import Optional, Dict, List, Tuple
from thefuzz import fuzz, process
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from tqdm.auto import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('categoricalapp.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SmartCategoricalCleaner:
    """Automated categorical data cleaner with performance optimizations"""
    
    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None, 
                n_jobs: int = -1, memory_efficient: bool = False):
        """
        Initialize the categorical data cleaner
        Args:
            df: Input DataFrame
            target_column: Target column name for supervised encodings
            n_jobs: Number of parallel jobs (-1 for all cores)
            memory_efficient: If True, operate on original dataframe to save memory
        """
        try:
            self.df = df if memory_efficient else df.copy()
            self.target = target_column
            self.report = {}
            self.cleaned_columns = []
            self.n_jobs = n_jobs
            self.memory_efficient = memory_efficient
            self._category_cache = {}
            self.execution_times = {}
            
            self._df_info = {
                'shape': df.shape,
                'memory': df.memory_usage(deep=True).sum() / (1024**2)  # in MB
            }
            
            logger.info(f"Initialized cleaner with dataframe shape {df.shape}")
        except Exception as e:
            logger.error(f"Failed to initialize cleaner: {str(e)}")
            raise
        
    def auto_clean(self, column: str, 
                  fix_typos: bool = True,
                  group_rare: bool = True,
                  rare_threshold: float = 0.05,
                  apply_encoding: bool = True,
                  encoding_strategy: Optional[str] = None,
                  create_features: bool = True,
                  similarity_threshold: float = 80) -> pd.DataFrame:
        """
        Main cleaning pipeline with customizable options
        
        Args:
            column: Name of column to clean
            fix_typos: Whether to automatically correct typos
            group_rare: Whether to group rare categories
            rare_threshold: Frequency below which categories are grouped
            apply_encoding: Whether to apply encoding
            encoding_strategy: Encoding method ('onehot', 'label', 'ordinal', 'frequency')
            create_features: Whether to generate derived features
            similarity_threshold: Threshold for typo detection (0-100)
            
        Returns:
            DataFrame with cleaned categorical column
        """
        start_time = time.time()
        
        try:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in dataset")
                
            logger.info(f"Starting cleaning for column: {column}")
            
            self._detect_patterns(column)
            
            self._handle_missing(column)
            self._standardize_text(column)
            
            if fix_typos:
                self._fix_typos(column, similarity_threshold=similarity_threshold)
                
            if group_rare:
                self._group_rare_categories(column, threshold=rare_threshold)
                
            if apply_encoding:
                self._optimize_encoding(column, strategy=encoding_strategy)
                
            if create_features:
                self._auto_feature_engineering(column)
            
            self.report[column]['after_stats'] = {
                'nunique': self.df[column].nunique() if column in self.df.columns else 0,
                'missing': self.df[column].isna().sum() if column in self.df.columns else 0
            }
            
            self.cleaned_columns.append(column)
            
            # Clean up 
            if column in self._category_cache:
                del self._category_cache[column]
            
            end_time = time.time()
            self.execution_times[column] = end_time - start_time
            logger.info(f"Finished cleaning {column} in {end_time - start_time:.2f} seconds")
            
            # garbage collection 
            if self.memory_efficient:
                gc.collect()
                
            return self.df
            
        except Exception as e:
            logger.error(f"Cleaning failed for column {column}: {str(e)}")
            raise

    def _detect_patterns(self, column: str):
        """
        Auto-detect data patterns and issues
        
        Identifies potential issues like missing values, rare categories,
        and possible typos in the specified column.
        """
        try:
            self.report[column] = {
                'unique_values': self.df[column].nunique(),
                'missing_values': self.df[column].isna().sum(),
                'top_categories': self.df[column].value_counts(normalize=True).nlargest(5).to_dict(),
                'typo_candidates': {},
                'rare_categories': [],
                'actions_performed': []
            }
            
            total_rows = len(self.df)
            cardinality_ratio = self.df[column].nunique() / total_rows
            self.report[column]['high_cardinality'] = cardinality_ratio > 0.5
        
            self.report[column]['typo_candidates'] = self._find_typo_candidates(column)
            self.report[column]['rare_categories'] = self._find_rare_categories(column)
            
        except Exception as e:
            logger.error(f"Error detecting patterns for {column}: {str(e)}")
            
            if column not in self.report:
                self.report[column] = {'actions_performed': []}

    def _find_typo_candidates(self, column: str, threshold: int = 80) -> Dict[str, str]:
        """
        Find potential typos using fuzzy string matching
        
        Args:
            column: Column name to check for typos
            threshold: Similarity threshold (0-100)
            
        Returns:
            Dictionary mapping potential typos to their likely corrections
        """
        try:
            if column not in self._category_cache:
                self._category_cache[column] = {
                    'top_cats': list(self.report[column]['top_categories'].keys()),
                    'unique_vals': set(self.df[column].dropna().unique())
                }
            
            top_cats = self._category_cache[column]['top_cats']
            unique_vals = self._category_cache[column]['unique_vals']
            
            if len(top_cats) < 2 or not pd.api.types.is_string_dtype(self.df[column]):
                return {}
            
            if len(unique_vals) > 1000 and self.n_jobs != 1:
                unique_list = list(unique_vals)
                batch_size = min(1000, len(unique_list))
                batches = [unique_list[i:i+batch_size] for i in range(0, len(unique_list), batch_size)]
                
                results = []
                for batch in batches:
                    batch_results = Parallel(n_jobs=self.n_jobs)(
                        delayed(self._process_typo_candidate)(val, top_cats, threshold) 
                        for val in batch if isinstance(val, str) and val not in top_cats
                    )
                    results.extend([r for r in batch_results if r])
                
                return {orig: match for orig, match in results if match}
                
            else:
                typo_candidates = {}
                for val in unique_vals:
                    if not isinstance(val, str) or val in top_cats:
                        continue
                        
                    match = process.extractOne(val, top_cats, scorer=fuzz.ratio)
                    if match and match[1] >= threshold:
                        typo_candidates[val] = match[0]
                
                return typo_candidates
                
        except Exception as e:
            logger.error(f"Error finding typos in {column}: {str(e)}")
            return {}
    
    def _process_typo_candidate(self, val, top_cats, threshold):
        """
        Helper function for parallel typo detection
        
        Args:
            val: Value to check
            top_cats: List of common categories to match against
            threshold: Similarity threshold
            
        Returns:
            Tuple of (original_value, corrected_value) if match found, None otherwise
        """
        try:
            if not isinstance(val, str) or val in top_cats:
                return None
                
            match = process.extractOne(val, top_cats, scorer=fuzz.ratio)
            if match and match[1] >= threshold:
                return (val, match[0])
            return None
        except:
            return None

    def _find_rare_categories(self, column: str, threshold: float = 0.05) -> List[str]:
        """
        Identify categories that occur below a frequency threshold
        
        Args:
            column: Column name to analyze
            threshold: Frequency threshold below which categories are considered rare
            
        Returns:
            List of category values that are considered rare
        """
        try:
            if len(self.df) < 20 or self.df[column].nunique() == len(self.df):
                return []
                
            freq = self.df[column].value_counts(normalize=True)
            rare_cats = list(freq[freq < threshold].index)
            
            return rare_cats
            
        except Exception as e:
            logger.error(f"Error finding rare categories in {column}: {str(e)}")
            return []

    def _handle_missing(self, column: str):
        """
        Auto-handle missing values in categorical columns
        
        Uses 'Unknown' for high-cardinality columns and mode imputation for others
        """
        try:
            missing_count = self.df[column].isna().sum()
            
            if missing_count > 0:
                if self.df[column].nunique() > 10 or len(self.df) < 100:
                    self.df[column] = self.df[column].fillna('Unknown')
                    method = "'Unknown' placeholder"
                else:
                    mode_value = self.df[column].mode()[0]
                    self.df[column] = self.df[column].fillna(mode_value)
                    method = f"mode imputation"
                    
                self.report[column]['actions_performed'].append(
                    f"Filled {missing_count} missing values using {method}"
                )
                
        except Exception as e:
            logger.error(f"Error handling missing values in {column}: {str(e)}")

    def _standardize_text(self, column: str):
        """
        Auto-format text data with optimized vectorization
        
        Converts to lowercase, removes extra spaces and standardizes formatting
        """
        try:
            if pd.api.types.is_string_dtype(self.df[column]) or self.df[column].dtype == 'object':
                non_na_mask = ~self.df[column].isna()
                
                if non_na_mask.any():
                    self.df.loc[non_na_mask, column] = (
                        self.df.loc[non_na_mask, column]
                        .astype(str)
                        .str.lower()
                        .str.strip()
                        .str.replace(r'\s+', ' ', regex=True)
                    )
                
                self.report[column]['actions_performed'].append("Standardized text formatting")
                
        except Exception as e:
            logger.error(f"Error standardizing text in {column}: {str(e)}")

    def _fix_typos(self, column: str, similarity_threshold: float = 80):
        """
        Apply automatic typo correction using fuzzy matching
        
        Args:
            column: Column name to clean
            similarity_threshold: Similarity score threshold (0-100)
        """
        try:
            typos = self._find_typo_candidates(column, threshold=int(similarity_threshold))
            
            if typos:
                self.df[column] = self.df[column].replace(typos)
                
                self.report[column]['actions_performed'].append(
                    f"Corrected {len(typos)} typos using fuzzy matching"
                )
                
        except Exception as e:
            logger.error(f"Error fixing typos in {column}: {str(e)}")

    def _group_rare_categories(self, column: str, threshold: float = 0.05):
        """
        Auto-group rare categories with optimized implementation
        
        Args:
            column: Column name to process
            threshold: Frequency threshold below which categories are grouped as 'Other'
        """
        try:
            rare = self._find_rare_categories(column, threshold)
            
            if rare:
                indicator_col = f"{column}_other"
                is_rare = self.df[column].isin(rare)
                self.df[indicator_col] = is_rare.astype(int)
                
                self.df.loc[is_rare, column] = 'Other'
                
                self.report[column]['actions_performed'].append(
                    f"Grouped {len(rare)} rare categories as 'Other'"
                )
                
        except Exception as e:
            logger.error(f"Error grouping rare categories in {column}: {str(e)}")

    def _optimize_encoding(self, column: str, strategy: Optional[str] = None):
        """
        Auto-select best encoding strategy for categorical columns
        
        Args:
            column: Column name to encode
            strategy: Encoding strategy ('onehot', 'label', 'ordinal', 'frequency')
                     If None, strategy is auto-selected based on cardinality
        """
        try:
            if column not in self.df.columns:
                return
                
            unique_count = self.df[column].nunique()
            
            if strategy is None:
                if unique_count <= 10:
                    strategy = 'onehot'
                elif unique_count <= 100 and self.target is not None:
                    strategy = 'label'
                else:
                    strategy = 'ordinal' if unique_count <= 1000 else 'frequency'
            
            if strategy == 'onehot':
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(self.df[[column]])
                feature_names = encoder.get_feature_names_out([column])
                
                encoded_df = pd.DataFrame(
                    encoded, 
                    columns=feature_names,
                    index=self.df.index
                )
                
                self.df = pd.concat([
                    self.df.drop(columns=[column]),
                    encoded_df
                ], axis=1)
                
                self.report[column]['actions_performed'].append(
                    f"Applied OneHot encoding, created {len(feature_names)} features"
                )
                
            elif strategy == 'label':
                value_mapping = {i: val for i, val in enumerate(
                    self.df[column].astype('category').cat.categories
                )}
                
                le = LabelEncoder()
                self.df[f"{column}_encoded"] = le.fit_transform(self.df[column].fillna('Unknown'))
                
                self.report[column]['encoding_map'] = value_mapping
                self.report[column]['actions_performed'].append("Applied Label encoding")
                
            elif strategy == 'ordinal':
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                self.df[f"{column}_ord"] = encoder.fit_transform(
                    self.df[[column]].fillna('Unknown')
                )
                
                self.report[column]['actions_performed'].append("Applied Ordinal encoding")
                
            elif strategy == 'frequency':
                freq_map = self.df[column].value_counts(normalize=True)
                self.df[f"{column}_freq"] = self.df[column].map(freq_map).fillna(0)
                
                self.report[column]['actions_performed'].append("Applied Frequency encoding")
                
        except Exception as e:
            logger.error(f"Error encoding {column}: {str(e)}")
            self.report[column]['actions_performed'].append(f"Encoding failed: {str(e)}")

    def _auto_feature_engineering(self, column: str):
        """
        Create derived features with optimized implementation
        
        Creates frequency encoding, target encoding (if target column available),
        and limited interaction features with previously cleaned columns.
        """
        try:
            if column not in self.df.columns:
                return
                
            # created features
            created_features = []
            
            # 1. Count encoding - frequency of each category
            freq_map = self.df[column].value_counts(normalize=True)
            self.df[f"{column}_freq"] = self.df[column].map(freq_map).fillna(0)
            created_features.append(f"{column}_freq")
            
            # 2. Target encoding if target column available
            if self.target is not None and self.target in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[self.target]):
                    target_means = self.df.groupby(column)[self.target].mean()
                    global_mean = self.df[self.target].mean()
                    
                    # Simple smoothing
                    n = self.df.groupby(column).size()
                    alpha = 10
                    smooth_means = (n * target_means + alpha * global_mean) / (n + alpha)
                    
                    self.df[f"{column}_target"] = self.df[column].map(smooth_means).fillna(global_mean)
                    created_features.append(f"{column}_target")
            
            # 3. Create limited interaction features
            if len(self.cleaned_columns) > 0:
                interaction_candidates = [
                    col for col in self.cleaned_columns 
                    if col != column and col in self.df.columns
                ][:1]  # Limit to just 1 cleaned column
                
                for other_col in interaction_candidates:
                    self.df[f"{column}_{other_col}_interact"] = (
                        self.df[column].astype(str) + "_" + self.df[other_col].astype(str)
                    )
                    created_features.append(f"{column}_{other_col}_interact")
            
            if created_features:
                self.report[column]['actions_performed'].append(
                    f"Created {len(created_features)} new features"
                )
                
        except Exception as e:
            logger.error(f"Error creating features for {column}: {str(e)}")

    def get_cleaning_report(self) -> Dict:
        """
        Generate detailed cleaning report with execution times
        
        Returns:
            Dictionary containing cleaning statistics and actions for each column
        """
        detailed_report = {}
        
        for col, info in self.report.items():
            detailed_report[col] = {
                'actions_performed': info.get('actions_performed', []),
                'before_stats': {
                    'unique_values': info.get('unique_values', 0),
                    'missing_values': info.get('missing_values', 0),
                },
                'after_stats': info.get('after_stats', {}),
                'execution_time': self.execution_times.get(col, 0)
            }
            
        return detailed_report


def clean_all_categorical_columns(df: pd.DataFrame, 
                              target_column: Optional[str] = None,
                              excluded_columns: List[str] = None,
                              n_jobs: int = -1,
                              memory_efficient: bool = False,
                              **kwargs) -> pd.DataFrame:
    """
    Clean all detected categorical columns in a dataframe
    
    Args:
        df: Input DataFrame
        target_column: Target column for supervised learning
        excluded_columns: Columns to exclude from cleaning
        n_jobs: Number of parallel jobs (-1 for all cores)
        memory_efficient: If True, operate in memory-efficient mode
        **kwargs: Additional arguments for auto_clean
        
    Returns:
        DataFrame with cleaned categorical columns
    """
    try:
        start_time = time.time()
        excluded = excluded_columns or []
        
        # Initialize cleaner
        cleaner = SmartCategoricalCleaner(df, target_column, n_jobs=n_jobs, memory_efficient=memory_efficient)
        
        # Auto-detect categorical columns
        categorical_columns = detect_categorical_columns(df)
        categorical_columns = [col for col in categorical_columns 
                              if col not in excluded and col != target_column]
        
        # Process each column with progress bar
        logger.info(f"Processing {len(categorical_columns)} categorical columns")
        for col in tqdm(categorical_columns, desc="Cleaning columns"):
            try:
                cleaner.auto_clean(col, **kwargs)
            except Exception as e:
                logger.error(f"Failed to clean {col}: {str(e)}")
        
        logger.info(f"Completed in {time.time() - start_time:.2f} seconds")
        return cleaner.df
        
    except Exception as e:
        logger.error(f"Error in categorical cleaning: {str(e)}")
        return df  # Return original dataframe if processing fails


def detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
     automatic detection for categorical columns
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of column names that likely represent categorical features
    """
    categorical_cols = []
    num_rows = len(df)
    
    for col in df.columns:
        # Skip columns with high percentage of missing values
        if df[col].isna().mean() > 0.9:
            continue
            
        # Any object/string column is considered categorical
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            categorical_cols.append(col)
                
        # Numeric columns with limited unique values are categorical
        elif pd.api.types.is_numeric_dtype(df[col]):
            unique_count = df[col].nunique()
            if unique_count <= min(20, num_rows * 0.5):  # More generous threshold
                categorical_cols.append(col)
    
    return categorical_cols


def save_cleaning_report_as_text(cleaner: SmartCategoricalCleaner, file_path: str = "cleaning_report.txt") -> None:
    """
    Save the categorical data cleaning report as a plain text file
    
    Args:
        cleaner: The SmartCategoricalCleaner instance with completed cleaning operations
        file_path: Path where to save the text report
        
    Returns:
        None
    """
    try:
        report = cleaner.get_cleaning_report()
        df_info = cleaner._df_info
        
        with open(file_path, 'w') as f:
            # Header
            f.write("==================================================\n")
            f.write("             CATEGORICAL CLEANING REPORT           \n")
            f.write("==================================================\n\n")
            
            # Dataset info
            f.write("DATASET INFORMATION:\n")
            f.write(f"Shape: {df_info['shape'][0]} rows x {df_info['shape'][1]} columns\n")
            f.write(f"Memory usage: {df_info['memory']:.2f} MB\n\n")
            
            # Total execution time
            total_time = sum(cleaner.execution_times.values())
            f.write(f"Total cleaning time: {total_time:.2f} seconds\n\n")
            
            # Column summaries
            f.write("==================================================\n")
            f.write("               COLUMN SUMMARIES                   \n")
            f.write("==================================================\n\n")
            
            for col, details in report.items():
                f.write(f"COLUMN: {col}\n")
                f.write("-" * 50 + "\n")
                
                # Before stats
                before_stats = details.get('before_stats', {})
                f.write(f"Before cleaning:\n")
                f.write(f"  - Unique values: {before_stats.get('unique_values', 'N/A')}\n")
                f.write(f"  - Missing values: {before_stats.get('missing_values', 'N/A')}\n")
                
                # After stats
                after_stats = details.get('after_stats', {})
                f.write(f"After cleaning:\n")
                f.write(f"  - Unique values: {after_stats.get('nunique', 'N/A')}\n")
                f.write(f"  - Missing values: {after_stats.get('missing', 'N/A')}\n")
                
                # Actions performed
                f.write("\nActions performed:\n")
                for i, action in enumerate(details.get('actions_performed', []), 1):
                    f.write(f"  {i}. {action}\n")
                
                # Execution time
                exec_time = details.get('execution_time', 0)
                f.write(f"\nExecution time: {exec_time:.2f} seconds\n\n")
                
                f.write("==================================================\n\n")
                
            f.write("End of report\n")
            
        logger.info(f"Cleaning report saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save cleaning report: {str(e)}")
        return False


def print_cleaning_summary(cleaner: SmartCategoricalCleaner) -> None:
    """
    Print a text-based summary of the categorical data cleaning to the console
    
    Args:
        cleaner: The SmartCategoricalCleaner instance with completed cleaning operations
        
    Returns:
        None
    """
    try:
        report = cleaner.get_cleaning_report()
        total_time = sum(cleaner.execution_times.values())
        
        print("\n" + "=" * 60)
        print(" " * 15 + "CATEGORICAL CLEANING SUMMARY")
        print("=" * 60)
        
        print(f"\nProcessed {len(report)} columns in {total_time:.2f} seconds")
        print(f"Columns processed: {', '.join(report.keys())}")
        
        print("\n" + "-" * 60)
        print("COLUMN HIGHLIGHTS")
        print("-" * 60)
        
        for col, details in report.items():
            before_missing = details.get('before_stats', {}).get('missing_values', 0)
            after_missing = details.get('after_stats', {}).get('missing', 0)
            
            actions = details.get('actions_performed', [])
            action_count = len(actions)
            
            print(f"\n{col}:")
            print(f"  • {action_count} cleaning actions performed")
            if before_missing > 0:
                missing_fixed = before_missing - after_missing
                print(f"  • {missing_fixed} missing values handled ({before_missing} → {after_missing})")
            
            print(f"  • Time: {details.get('execution_time', 0):.2f}s")
        
        print("\nFor detailed information, see the full report file.")
        
    except Exception as e:
        logger.error(f"Failed to print cleaning summary: {str(e)}")


def clean_and_report(df: pd.DataFrame, 
                    target_column: Optional[str] = None,
                    excluded_columns: List[str] = None,
                    report_path: str = "cleaning_report.txt",
                    print_summary: bool = True,
                    **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean categorical columns and generate a comprehensive text report
    
    Args:
        df: Input DataFrame
        target_column: Target column for supervised learning
        excluded_columns: Columns to exclude from cleaning
        report_path: Path to save the text report
        print_summary: Whether to print a summary to console
        **kwargs: Additional arguments for the cleaner
        
    Returns:
        Tuple containing the cleaned DataFrame and cleaning report dict
    """
    try:
        start_time = time.time()
        excluded = excluded_columns or []
        
        # Initialize cleaner
        cleaner = SmartCategoricalCleaner(df, target_column, 
                                        n_jobs=kwargs.get('n_jobs', -1), 
                                        memory_efficient=kwargs.get('memory_efficient', False))
        
        # Auto-detect categorical columns
        categorical_columns = detect_categorical_columns(df)
        categorical_columns = [col for col in categorical_columns 
                              if col not in excluded and col != target_column]
        
        # Process each column with progress bar
        logger.info(f"Processing {len(categorical_columns)} categorical columns")
        for col in tqdm(categorical_columns, desc="Cleaning columns"):
            try:
                cleaner.auto_clean(col, **kwargs)
            except Exception as e:
                logger.error(f"Failed to clean {col}: {str(e)}")
        
        # Generate report
        if report_path:
            save_cleaning_report_as_text(cleaner, report_path)
            
        if print_summary:
            print_cleaning_summary(cleaner)
            
        logger.info(f"Completed in {time.time() - start_time:.2f} seconds")
        
        return cleaner.df, cleaner.get_cleaning_report()
        
    except Exception as e:
        logger.error(f"Error in categorical cleaning: {str(e)}")
        return df, {}  # Return original dataframe if processing fails


def detect_temporal_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect columns that might represent temporal data (year, month, etc.)
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of column names that likely represent temporal features
    """
    temporal_columns = []
    
    time_related_names = [
        'year', 'month', 'day', 'week', 'quarter', 'semester',
        'yr', 'mo', 'season', 'period', 'fiscal'
    ]
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check if column name contains temporal terms
        if any(term in col_lower for term in time_related_names):
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check for reasonable ranges for year, month, day, etc.
                if 'year' in col_lower or 'yr' in col_lower:
                    if df[col].min() >= 1900 and df[col].max() <= 2100:
                        temporal_columns.append(col)
                elif 'month' in col_lower or 'mo' in col_lower:
                    if df[col].min() >= 1 and df[col].max() <= 12:
                        temporal_columns.append(col)
                elif 'day' in col_lower:
                    if df[col].min() >= 1 and df[col].max() <= 31:
                        temporal_columns.append(col)
                elif 'quarter' in col_lower:
                    if df[col].min() >= 1 and df[col].max() <= 4:
                        temporal_columns.append(col)
                else:
                    # If we can't determine type but name suggests temporal
                    temporal_columns.append(col)
    
    return temporal_columns
