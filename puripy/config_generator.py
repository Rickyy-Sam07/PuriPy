import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union, List, Callable
from functools import partial
import shutil
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

# --------------------------
# Pure functions (no side effects)
# --------------------------

def load_data(source: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Load data from source (path or DataFrame)"""
    if isinstance(source, str):
        return pd.read_csv(source)
    if isinstance(source, pd.DataFrame):
        return source.copy()
    raise ValueError("Source must be DataFrame or file path")

def detect_partial_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Identify columns with some numeric values"""
    numeric_like = []
    for col in df.select_dtypes(include=['object']).columns:
        try:
            if pd.to_numeric(df[col], errors='coerce').notna().mean() > 0.5:
                numeric_like.append(col)
        except:
            continue
    return numeric_like

def create_constraint(min_val: float, max_val: float) -> Callable:
    """Create range constraint with captured values"""
    return lambda x: x.between(min_val, max_val)

# --------------------------
# Configuration builders (pure where possible)
# --------------------------

def build_type_conversion_config(df: pd.DataFrame) -> Dict:
    numeric_like = detect_partial_numeric_columns(df)
    return {'type_conversion': {'numeric_cols': numeric_like}} if numeric_like else {}

def build_missing_values_config(df: pd.DataFrame) -> Dict:
    return {'missing_values': {'strategy': 'mean', 'threshold': 0.5}} if df.isna().any().any() else {}

def build_outlier_config(df: pd.DataFrame) -> Dict:
    numeric_cols = df.select_dtypes('number').columns.tolist()
    return {'outliers': {'method': 'iqr', 'action': 'cap'}} if numeric_cols else {}

def build_duplicates_config(df: pd.DataFrame) -> Dict:
    has_dupes = df.duplicated().any()
    return {'duplicates': {'subset': None, 'keep': 'first'}} if has_dupes else {}

def build_data_errors_config(df: pd.DataFrame) -> Dict:
    constraints = {}
    if 'age' in df.columns:
        constraints['age'] = lambda x: x > 0
    if 'score' in df.columns:
        constraints['score'] = lambda x: (x >= 0) & (x <= 100)
    return {'data_errors': {'constraints': constraints, 'correction': 'median'}} if constraints else {}

def build_precision_config(df: pd.DataFrame) -> Dict:
    float_cols = df.select_dtypes('float').columns.tolist()
    return {'precision': {col: 2 for col in float_cols}} if float_cols else {}

# --------------------------
# Main composition
# --------------------------

def generate_config(source: Union[str, pd.DataFrame]) -> Dict:
    """Generate cleaning configuration from data source"""
    df = load_data(source)
    logger.info(f"Analyzing dataset with shape {df.shape}")
    
    config_builders = [
        build_type_conversion_config,
        build_missing_values_config,
        build_outlier_config,
        build_duplicates_config,
        build_data_errors_config,
        build_precision_config
    ]
    
    return {
        k: v for builder in config_builders
        for k, v in builder(df).items()
    }

def copy_default_config(destination=None):
    """
    Copy the default config.py from the installed package to the user's current directory (or specified destination).
    """
    package_dir = os.path.dirname(__file__)
    src = os.path.join(package_dir, 'config.py')
    if destination is None:
        destination = os.path.join(os.getcwd(), 'config.py')
    else:
        destination = os.path.abspath(destination)
    if os.path.exists(destination):
        print(f"config.py already exists at {destination}. Aborting to avoid overwrite.")
        return
    shutil.copyfile(src, destination)
    print(f"Default config.py copied to {destination}")

# --------------------------
# Example usage
# --------------------------
'''
if __name__ == "__main__":
    data = pd.DataFrame({
        'age': [25, 30, 22, 40, 22, 30],
        'salary': [50000, 60000, 52000, 70000, None, 60000],
        'department_id': ['101', '102', '103', '104', '105', '106'],
        'city': ['NYC', 'LA', 'NYC', 'LA', 'Chicago', 'NYC']
    })
    
    config = generate_config(data)
    print("Generated Configuration:")
    print(config)
    '''

if __name__ == "__main__":
    copy_default_config()
