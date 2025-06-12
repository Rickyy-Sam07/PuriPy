from setuptools import setup, find_packages

setup(
    name='CleansiPy',
    version='0.1.0',
    description='a modular Python package for cleaning text, categorical, numerical, and datetime data. It offers configurable pipelines with support for preprocessing, typo correction, encoding, imputation, logging, parallel processing, and audit reporting—perfect for data scientists handling messy, real-world datasets in ML workflows.',
    author='Sambhranta Ghosh',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Core data processing
        "pandas",
        "numpy",
        # NLP & Text Processing
        "nltk",
        "emoji",
        "contractions",
        "better-profanity",
        "textblob",
        # String matching
        "thefuzz",
        "python-Levenshtein",
        # Machine learning
        "scikit-learn",
        "statsmodels",
        # Date & Time handling
        "pytz",
        "python-dateutil",
        # Progress bars & parallel processing
        "tqdm",
        "joblib",
        # Optional but recommended (for advanced users)
        # "matplotlib",
        # "seaborn",
        # Development dependencies (for contributors only)
        # "pytest",
        # "black",
        # "flake8",
    ],
    entry_points={
        'console_scripts': [
            'cleansipy=puripy.app:main',
        ],
    },
    package_data={
        # The package directory is still 'puripy', but the PyPI name is 'CleansiPy'
        'puripy': ['assets/*', 'config.py']
    },
    # license="MIT",  # Temporarily remove to avoid license-file metadata issue
    url="https://github.com/Rickyy-Sam07/CleansiPy.git",
)
