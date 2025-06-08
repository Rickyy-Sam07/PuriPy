# categoricaldata.py - Optimized with Thefuzz library
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
    handlers=[logging.FileHandler('auto_cleaner.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SmartCategoricalCleaner:
    """Automated categorical data cleaner with performance optimizations"""
    
    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None, 
                n_jobs: int = -1, memory_efficient: bool = False):
        try:
            # Make a copy only if memory_efficient is False
            self.df = df if memory_efficient else df.copy()
            self.target = target_column
            self.report = {}
            self.cleaned_columns = []
            self.n_jobs = n_jobs
            self.memory_efficient = memory_efficient
            self._category_cache = {}
            self.execution_times = {}
            
            # Store DataFrame info
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
        """Main cleaning pipeline with customizable options"""
        start_time = time.time()
        
        try:
            # Validate column exists
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in dataset")
                
            logger.info(f"Starting cleaning for column: {column}")
            
            # Store original state for report
            self._detect_patterns(column)
            
            # Apply cleaning steps
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
            
            # Update report with after stats
            self.report[column]['after_stats'] = {
                'nunique': self.df[column].nunique() if column in self.df.columns else 0,
                'missing': self.df[column].isna().sum() if column in self.df.columns else 0
            }
            
            self.cleaned_columns.append(column)
            
            # Clean up cache to free memory
            if column in self._category_cache:
                del self._category_cache[column]
            
            # Record execution time
            end_time = time.time()
            self.execution_times[column] = end_time - start_time
            logger.info(f"Finished cleaning {column} in {end_time - start_time:.2f} seconds")
            
            # Run garbage collection if in memory-efficient mode
            if self.memory_efficient:
                gc.collect()
                
            return self.df
            
        except Exception as e:
            logger.error(f"Cleaning failed for column {column}: {str(e)}")
            raise

    def _detect_patterns(self, column: str):
        """Auto-detect data patterns and issues"""
        try:
            # Initialize report for this column
            self.report[column] = {
                'unique_values': self.df[column].nunique(),
                'missing_values': self.df[column].isna().sum(),
                'top_categories': self.df[column].value_counts(normalize=True).nlargest(5).to_dict(),
                'typo_candidates': {},
                'rare_categories': [],
                'actions_performed': []
            }
            
            # Basic pattern detection
            total_rows = len(self.df)
            cardinality_ratio = self.df[column].nunique() / total_rows
            self.report[column]['high_cardinality'] = cardinality_ratio > 0.5
            
            # Find typos and rare categories
            self.report[column]['typo_candidates'] = self._find_typo_candidates(column)
            self.report[column]['rare_categories'] = self._find_rare_categories(column)
            
        except Exception as e:
            logger.error(f"Error detecting patterns for {column}: {str(e)}")
            # Initialize empty report to avoid errors in other methods
            if column not in self.report:
                self.report[column] = {'actions_performed': []}

    def _find_typo_candidates(self, column: str, threshold: int = 80) -> Dict[str, str]:
        """Find potential typos using Thefuzz library"""
        try:
            # Cache the top categories for this column
            if column not in self._category_cache:
                self._category_cache[column] = {
                    'top_cats': list(self.report[column]['top_categories'].keys()),
                    'unique_vals': set(self.df[column].dropna().unique())
                }
            
            top_cats = self._category_cache[column]['top_cats']
            unique_vals = self._category_cache[column]['unique_vals']
            
            # Skip if conditions aren't right
            if len(top_cats) < 2 or not pd.api.types.is_string_dtype(self.df[column]):
                return {}
            
            # Process large datasets in parallel
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
                # Serial processing for smaller datasets
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
        """Helper function for parallel typo detection"""
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
        """Identify categories below frequency threshold"""
        try:
            # Skip if not enough data or all values are unique
            if len(self.df) < 20 or self.df[column].nunique() == len(self.df):
                return []
                
            # Use vectorized operations for better performance
            freq = self.df[column].value_counts(normalize=True)
            rare_cats = list(freq[freq < threshold].index)
            
            return rare_cats
            
        except Exception as e:
            logger.error(f"Error finding rare categories in {column}: {str(e)}")
            return []

    def _handle_missing(self, column: str):
        """Auto-handle missing values"""
        try:
            missing_count = self.df[column].isna().sum()
            
            if missing_count > 0:
                if self.df[column].nunique() > 10 or len(self.df) < 100:
                    # For high cardinality or small datasets, use 'Unknown'
                    self.df[column] = self.df[column].fillna('Unknown')
                    method = "'Unknown' placeholder"
                else:
                    # For low cardinality with sufficient data, use mode
                    mode_value = self.df[column].mode()[0]
                    self.df[column] = self.df[column].fillna(mode_value)
                    method = f"mode imputation"
                    
                self.report[column]['actions_performed'].append(
                    f"Filled {missing_count} missing values using {method}"
                )
                
        except Exception as e:
            logger.error(f"Error handling missing values in {column}: {str(e)}")

    def _standardize_text(self, column: str):
        """Auto-format text data with optimized vectorization"""
        try:
            # Only process if column contains string data
            if pd.api.types.is_string_dtype(self.df[column]) or self.df[column].dtype == 'object':
                # Get non-NaN mask once for efficiency
                non_na_mask = ~self.df[column].isna()
                
                if non_na_mask.any():
                    # Convert only non-NaN values to string and apply transformations
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
        """Apply automatic typo correction using Thefuzz"""
        try:
            # Get typos with specified threshold
            typos = self._find_typo_candidates(column, threshold=int(similarity_threshold))
            
            if typos:
                # Use efficient replacement
                self.df[column] = self.df[column].replace(typos)
                
                self.report[column]['actions_performed'].append(
                    f"Corrected {len(typos)} typos using fuzzy matching"
                )
                
        except Exception as e:
            logger.error(f"Error fixing typos in {column}: {str(e)}")

    def _group_rare_categories(self, column: str, threshold: float = 0.05):
        """Auto-group rare categories with optimized implementation"""
        try:
            # Find rare categories
            rare = self._find_rare_categories(column, threshold)
            
            if rare:
                # Create indicator column to track which rows were grouped
                indicator_col = f"{column}_other"
                is_rare = self.df[column].isin(rare)
                self.df[indicator_col] = is_rare.astype(int)
                
                # Group rare categories
                self.df.loc[is_rare, column] = 'Other'
                
                self.report[column]['actions_performed'].append(
                    f"Grouped {len(rare)} rare categories as 'Other'"
                )
                
        except Exception as e:
            logger.error(f"Error grouping rare categories in {column}: {str(e)}")

    def _optimize_encoding(self, column: str, strategy: Optional[str] = None):
        """Auto-select best encoding strategy"""
        try:
            if column not in self.df.columns:
                return
                
            unique_count = self.df[column].nunique()
            
            # Auto-select strategy based on cardinality
            if strategy is None:
                if unique_count <= 10:
                    strategy = 'onehot'
                elif unique_count <= 100 and self.target is not None:
                    strategy = 'label'
                else:
                    strategy = 'ordinal' if unique_count <= 1000 else 'frequency'
            
            # Apply encoding strategy
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
        """Create derived features with optimized implementation"""
        try:
            # Skip if column was encoded and no longer exists
            if column not in self.df.columns:
                return
                
            # Track created features
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
        """Generate detailed cleaning report with execution times"""
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

# Function to clean all categorical columns with progress bar
def clean_all_categorical_columns(df: pd.DataFrame, 
                              target_column: Optional[str] = None,
                              excluded_columns: List[str] = None,
                              n_jobs: int = -1,
                              memory_efficient: bool = False,
                              **kwargs) -> pd.DataFrame:
    """Clean all detected categorical columns in a dataframe"""
    try:
        start_time = time.time()
        excluded = excluded_columns or []
        
        # Initialize cleaner
        cleaner = SmartCategoricalCleaner(df, target_column, n_jobs=n_jobs, memory_efficient=memory_efficient)
        
        # Auto-detect categorical columns
        categorical_columns = [
            col for col in df.columns
            if col not in excluded and col != target_column and (
                pd.api.types.is_object_dtype(df[col]) or (
                    pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() < 20
                )
            )
        ]
        
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
    """Improved automatic detection for categorical columns"""
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
