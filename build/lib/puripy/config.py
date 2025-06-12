#############################################
#config2 for categoricaldata                   #
#############################################
# EDIT THESE SETTINGS AS NEEDED
config2 = {
    "INPUT_FILE": r"testdata\xx.csv",              # Path to your input CSV file
    "OUTPUT_FILE": r"testdata\cleaned.csv",        # Where to save the cleaned data
    "TARGET_COLUMN": None,               # Optional target column for supervised learning
    "COLUMNS_TO_CLEAN": ["category", "other_cat_col"],  # Set to None for auto-detection or list specific columns
    "EXCLUDE_COLUMNS": [],               # Columns to exclude from cleaning
    "EXPLORE_ONLY": False,               # Set to True to only explore data without cleaning
    "FILE_PATH": r"testdata\cleaning_report.txt",  # Path to save the cleaning report
    
    # Advanced Settings
    "FIX_TYPOS": True,                   # Whether to fix typos in text
    "GROUP_RARE": True,                  # Whether to group rare categories
    "RARE_THRESHOLD": 0.05,              # Categories with frequency below this are considered rare
    "SIMILARITY_THRESHOLD": 80,          # Threshold (0-100) for fuzzy matching of typos
    "MEMORY_EFFICIENT": False,           # Set to True for very large datasets
    "PARALLEL_JOBS": -1                  # Number of parallel jobs (-1 for all cores)
}
#############################################

# NUMERIC DATA CONFIG-
# --------------------------
# Modify these settings based on your specific dataset requirements.This configuration controls how the data cleaning pipeline processes your dataset.

DEFAULT_CONFIG = {

    'input_file': r'testdata\data.csv',  # Path to your input CSV file
    'output_file': r'testdata\cleaned_output.csv',  # Path for output cleaned CSV file
    'report_file': r'testdata\textreport.txt',  # Path to save cleaning reportq

    # Type Conversion Settings
    # -----------------------
    # Specify which columns should be converted to numeric types
    # For non-numeric datasets, set this to [] or remove columns that aren't numeric
    'type_conversion': {
        'numeric_cols': ['Sales_Before', 'Sales_After', 'Customer_Satisfaction_Before', 'Customer_Satisfaction_After']
    },
    
    # Missing Value Handling
    # ---------------------
    # strategy: how to fill missing values ('mean', 'median', 'mode')
    # threshold: maximum ratio of missing values allowed (0.0 to 1.0)
    'missing_values': {
        'strategy': 'mean',  # Options: 'mean', 'median', 'mode'
        'threshold': 0.5     # Columns with >40% missing values will be flagged
    },
    
    # Data Constraints & Validation
    # ----------------------------
    # Define valid ranges/rules for each column using lambda functions
    # correction: how to replace invalid values ('median', 'mean', 'mode')
    'data_errors': {
        'constraints': {
            'Sales_Before': lambda x: (x >= 50) & (x <= 500),  # Valid sales range
            'Sales_After': lambda x: (x >= 50) & (x <= 700),  # Valid sales range
            'Customer_Satisfaction_Before': lambda x: (x >= 0) & (x <= 100),  # Percentage-based score
            'Customer_Satisfaction_After': lambda x: (x >= 0) & (x <= 100)  # Percentage-based score
        },
        'correction': 'median'  # Use median of valid values to replace invalid ones
    },
    
    # Outlier Detection & Handling
    # --------------------------
    # method: technique to detect outliers ('iqr', 'zscore')
    # action: how to handle outliers ('cap', 'remove')
    # columns: specific columns to check for outliers
    'outliers': {
        'method': 'iqr',  # Interquartile Range method (Q1-1.5*IQR to Q3+1.5*IQR)
        'action': 'cap',  # Cap values at the boundaries instead of removing rows
        'columns': ['Sales_Before', 'Sales_After', 'Customer_Satisfaction_Before', 'Customer_Satisfaction_After']  # Columns to check
    },
    
    # Duplicate Handling
    # -----------------
    # subset: columns to consider when identifying duplicates (None = all columns)
    # keep: which occurrence to keep ('first', 'last', False)
    'duplicates': {
        'subset': None,  # Consider all columns when identifying duplicates
        'keep': 'first'  # Keep the first occurrence and remove others
    },
    
    # Numeric Precision
    # ----------------
    # Control decimal places for each column (0 = integer, >0 = decimal places)
    'precision': {
        'Sales_Before': 2,      # Two decimal places for currency
        'Sales_After': 2,       # Two decimal places for currency
        'Customer_Satisfaction_Before': 1,  # One decimal place for satisfaction scores
        'Customer_Satisfaction_After': 1    # One decimal place for satisfaction scores
    }
}


#############################################
#   date&time cleaner                 #
#############################################
# EDIT THESE SETTINGS AS NEEDED
config3 = {
    "INPUT_FILE": r"testdata\dates.csv",             # Path to your input CSV file
    "OUTPUT_FILE": r"testdata\cleaned.csv",          # Where to save the cleaned data
    "REPORT_FILE": r"testdata\date_cleaning_report.txt",  # Path to save the cleaning report

    # Date column settings
    "DATE_COLUMNS": None,                  # Set to None for auto-detection or list specific columns
                                           # Example: ["order_date", "shipping_date"]
    "START_END_PAIRS": [],                 # List of (start_date, end_date) column pairs to validate
                                           # Example: [("start_date", "end_date")]

    # Cleaning options
    "PARSE_DATES": True,                   # Convert strings to datetime objects
    "IMPUTE_MISSING": True,                # Fill in missing date values
    "STANDARDIZE_TIMEZONE": False,         # Convert to a standard timezone
    "EXTRACT_FEATURES": True,              # Generate calendar features from dates

    # Advanced settings
    "IMPUTATION_METHOD": "linear",         # Options: "linear", "forward", "backward", "seasonal", "mode"
    "SEASONAL_PERIOD": 7,                  # For seasonal imputation (e.g., 7=weekly, 12=monthly)
    "TARGET_TIMEZONE": "UTC",              # Target timezone for standardization
    "CONSISTENCY_ACTION": "flag",          # How to handle inconsistent dates: "flag", "swap", "drop", "truncate"
    "MEMORY_EFFICIENT": False,             # Set to True for very large datasets

    # Feature extraction
    "CALENDAR_FEATURES": [                 # Features to extract from date columns
        "year", "month", "day", "dayofweek", "quarter", 
        "is_weekend", "is_month_end", "is_quarter_end", 
        "fiscal_quarter", "season"
    ],
    "FISCAL_YEAR_START": 10                # Starting month of fiscal year (1-12)
}
#############################################
#==========================================
#TEXT CLEANING DATA CONFIGURATION
#======================================

config = {
    'lowercase': True,              # Convert all text to lowercase
    'remove_punctuation': True,     # Remove all punctuation marks
    'remove_stopwords': True,       # Remove common stopwords (e.g., "the", "is", "and")
    'remove_urls': True,            # Remove URLs and web addresses
    'remove_html': True,            # Remove HTML tags from text
    'remove_emojis': True,          # Remove emojis and special symbols
    'remove_numbers': True,         # Remove all numeric digits
    'expand_contractions': False,    # Expand contractions (e.g., "don't" -> "do not")
    'spelling_correction': False,    # Correct spelling mistakes in words
    'lemmatize': True,              # Reduce words to their base form (e.g., "running" -> "run")
    'stem': False,                  # Reduce words to their root form (e.g., "running" -> "run"); set True to enable
    'tokenize': 'word',             # Tokenize text into words ('word'), sentences ('sentence'), or None for no tokenization
    'ngram_range': (1, 1),          # Generate n-grams; (1,1) for unigrams only, (1,2) for unigrams and bigrams, etc.
    'profanity_filter': False,      # Remove or mask profane words; set True to enable
    'language': 'english',          # Language for stopwords, lemmatization, and spell checking
    'custom_stopwords': None,       # List of additional stopwords to remove (e.g., ['foo', 'bar']); None for default
    'custom_profanity': None,       # List of additional profane words to filter; None for default
    'input_file': r"testdata\test.csv",   # Path to input CSV file
    'output_file': r"testdata\cleaned_text.csv",  # Path to save cleaned output
    'report_file': r"testdata\text_cleaning_report.txt",  # Path to save cleaning report
    'text_column': None,            # Specify text column to clean (None for auto-detect)
    'sample_count': 5               # Number of samples to show in report
}