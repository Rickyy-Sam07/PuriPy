from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='CleansiPy',
    version='0.1.4',
    description='a modular Python package for cleaning text, categorical, numerical, and datetime data. It offers configurable pipelines with support for preprocessing, typo correction, encoding, imputation, logging, parallel processing, and audit reporting—perfect for data scientists handling messy, real-world datasets in ML workflows.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Sambhranta Ghosh',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "nltk",
        "emoji",
        "contractions",
        "better-profanity",
        "textblob",
        "thefuzz",
        "python-Levenshtein",
        "scikit-learn",
        "statsmodels",
        "pytz",
        "python-dateutil",
        "tqdm",
        "joblib"
    ],
    entry_points={
        'console_scripts': [
            'cleansipy=puripy.app:main',
            'cleansipy-config=puripy.config_generator:copy_default_config',
        ],
    },
    package_data={
        'CleansiPy': ['assets/*', 'config.py']
    },
    # license="MIT",  # Temporarily remove to avoid license-file metadata issue
    url="https://github.com/Rickyy-Sam07/CleansiPy.git",
)
