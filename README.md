

![logo](https://github.com/user-attachments/assets/9851a803-cc05-43b6-918e-e8a407d3296d)

DOWNLOADS : [![PyPI Downloads](https://static.pepy.tech/badge/cleansipy)](https://pepy.tech/projects/cleansipy)
# CleansiPy
CleansiPy 🧼📊
Clean your data like a pro — Text, Categorical, Numerical, and DateTime — all in one package.


🚀 Overview
CleansiPy is an all-in-one Python package designed to clean and preprocess messy datasets with ease and flexibility. It supports four major data types:

📝 Text – tokenization, stemming, lemmatization, stopword removal, n-gram generation, profanity filtering, emoji & HTML cleaning, and more.

🧮 Numerical – missing value handling, outlier detection, precision adjustment, type correction, and logging.

🧾 Categorical – typo correction, standardization, rare value grouping, encoding (OneHot, Label, Ordinal), and fuzzy matching.

🕒 DateTime – flexible parsing, timezone unification, feature extraction (day, month, weekday, etc.), imputation, and validation.

It’s built for data scientists, ML engineers, and analysts working on real-world data pipelines.

🔧 Installation
bash
Copy
Edit
pip install cleansipy
📦 Features
✅ Configurable, modular pipelines
✅ Works with pandas DataFrames
✅ Multi-core processing for speed
✅ NLTK/TextBlob integration for NLP
✅ sklearn support for encoding
✅ Detailed logs and cleaning reports
✅ Auto column detection
✅ Type-safe and test-friendly design

## ⚡ Quick Start

>> before doing any thing to avoid errors create a venv:
```powershell
python -m venv venv
```

-now activate the venv 
```powershell
venv\Scripts\activate
```


1. **Set up your configuration:**
   
   After installing, run the following command to copy the default config.py to your project directory:
   ```powershell
   cleansipy-config
   ```
   Then edit `config.py` to set your input/output file paths and other options before running the application.

2. **write the main file:**

```powershell
from puripy.app import main
if __name__ == "__main__":
    main()
```

3. **Run the application:**
   
   ```powershell
   python main.py
   ```
  

---

## 🖼️ Logo

The official Puripy logo is included in the package at `CleansiPy/assets/logo.png`.

To access or display the logo programmatically:

```python
from CleansiPy import get_logo_path, show_logo
print(get_logo_path())
show_logo()
```

---

## 📦 Package Structure

```
CleansiPy/
    __init__.py
    __main__.py
    app.py
    mainnum.py
    maincat.py
    maintext.py
    maindt.py
    logo.py
    config.py
   numericdata.py
   categoricaldata.py
   dt.py
   textualdata.py
    assets/
        logo.png
        README.txt
setup.py
requirements.txt
README.md
```

- All main code is inside the `CleansiPy/` directory for packaging.
- The logo is in `CleansiPy/assets/logo.png` and accessible via `get_logo_path()` and `show_logo()`.
- To run the app: set up config, install requirements, then run `python -m CleansiPy.app`.

---

in case you dont get the config.py then use this template : 

 ```powershell
   # ==================== CATEGORICAL DATA CONFIG ====================
config2 = {
    # File paths
    "INPUT_FILE": r"testdata\xx.csv",              # Raw data source
    "OUTPUT_FILE": r"testdata\cleaned.csv",        # Cleaned data destination
    "TARGET_COLUMN": None,                         # Target variable for ML tasks
    "FILE_PATH": r"testdata\cleaning_report.txt",  # Cleaning audit log
    
    # Column selection
    "COLUMNS_TO_CLEAN": ["category", "other_cat_col"],  # Specific columns to process
    "EXCLUDE_COLUMNS": [],                         # Columns to skip
    
    # Core cleaning features
    "FIX_TYPOS": True,                            # Auto-correct spelling variations
    "GROUP_RARE": True,                           # Consolidate infrequent categories
    "RARE_THRESHOLD": 0.05,                       # Minimum frequency to keep as separate category
    "SIMILARITY_THRESHOLD": 80,                   # Fuzzy matching sensitivity (0-100)
    
    # Performance
    "MEMORY_EFFICIENT": False,                    # Optimize for large datasets
    "PARALLEL_JOBS": -1                           # CPU cores to use (-1 = all)
}

# ==================== NUMERICAL DATA CONFIG ====================
DEFAULT_CONFIG = {
    # File handling
    'input_file': r'testdata\data.csv',
    'output_file': r'testdata\cleaned_output.csv',
    'report_file': r'testdata\textreport.txt',

    # Data type handling
    'type_conversion': {
        'numeric_cols': ['Sales_Before', 'Sales_After']  # Columns to force-convert to numeric
    },
    
    # Missing data
    'missing_values': {
        'strategy': 'mean',                      # Imputation method (mean/median/mode)
        'threshold': 0.5                         # Max allowed missingness per column
    },
    
    # Data validation
    'data_errors': {
        'constraints': {                         # Value range rules
            'Sales_Before': lambda x: (x >= 50) & (x <= 500)
        },
        'correction': 'median'                   # How to fix invalid values
    },
    
    # Outlier treatment
    'outliers': {
        'method': 'iqr',                         # Detection method (iqr/zscore)
        'action': 'cap',                         # Handling (cap/remove)
        'columns': ['Sales_Before', 'Sales_After']  # Columns to check
    },
    
    # Precision control
    'precision': {
        'Sales_Before': 2                        # Decimal places to round
    }
}

# ==================== DATE/TIME CONFIG ====================
config3 = {
    # File setup
    "INPUT_FILE": r"testdata\dates.csv",
    "OUTPUT_FILE": r"testdata\cleaned.csv",
    "REPORT_FILE": r"testdata\date_cleaning_report.txt",

    # Date processing
    "PARSE_DATES": True,                        # Convert string dates
    "IMPUTE_MISSING": True,                     # Fill missing timestamps
    "IMPUTATION_METHOD": "linear",              # Filling strategy
    "STANDARDIZE_TIMEZONE": False,              # Convert to target timezone
    
    # Feature generation
    "EXTRACT_FEATURES": True,                   # Create calendar features
    "CALENDAR_FEATURES": ["year", "month"],     # Features to extract
    "FISCAL_YEAR_START": 10                     # Fiscal year starting month
}

# ==================== TEXT CLEANING CONFIG ====================
config = {
    # Core text processing
    'lowercase': True,                          # Convert to lowercase
    'remove_punctuation': True,                 # Strip punctuation
    'remove_stopwords': True,                   # Filter common words
    'lemmatize': True,                          # Reduce to base form
    
    # Advanced cleaning
    'remove_urls': True,                        # Strip web addresses
    'remove_emojis': True,                      # Filter emoji characters
    'spelling_correction': False,               # Fix spelling errors
    
    # File handling
    'input_file': r"testdata\test.csv",
    'output_file': r"testdata\cleaned_text.csv",
    'text_column': None                         # Auto-detect text column
}
   ```

>> Author
Developed by Sambhranta Ghosh
Open to contributions, feedback, and improvements!  

For more, see the in-code docstrings and comments
visit this repo for more info and contributions : https://github.com/Rickyy-Sam07/CleansiPy.git
