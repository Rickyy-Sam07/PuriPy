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
pip install puripy
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

1. **Set up your configuration:**
   
   Edit `puripy/config.py` to set your input/output file paths and other options before running the application.

2. **Install requirements:**
   
   ```powershell
   pip install -r requirements.txt
   ```
   Or, if you want to use the package mode:
   ```powershell
   pip install .
   ```

3. **Run the application:**
   
   ```powershell
   python -m puripy.app
   ```
   Or, if you installed as a package and set up entry points:
   ```powershell
   puripy
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

>> Author
Developed by Sambhranta Ghosh
Open to contributions, feedback, and improvements!  

For more, see the in-code docstrings and comments
