# **Stop-word-behavior** by `Duc Vo` & `Roope Tukiainen`

## Abstract


This project evaluates stopword behavior across three distinct, categorized corpora: two English (NLTK Reuters, NPS Chat) and one Vietnamese (a balanced news dataset) . We identify potential stopwords using two primary methods: **Term Frequency-Inverse Document Frequency (TF-IDF)** and the [**Armano et al. (φ/δ) metrics**](https://www.semanticscholar.org/paper/Stopwords-Identification-by-Means-of-Characteristic-Armano-Fanni/42d58221ee7a0ce4ee8c8ddfe9d1b6b5fb29dd2c). The resulting candidate lists are evaluated by measuring their precision overlap against their respective default NLTK stopword lists . Finally, we analyze the statistical versus semantic coherence of these stopword sets by applying K-Means Clustering and t-SNE visualization to both their statistical φ/δ scores and their semantic embeddings (e.g., `Word2Vec`, `FastText`, `Glove`, and `PhoBERT`)


## `Duc Vo` - Analysis of Stopword Behavior in a Vietnamese Text Corpus

This repository contains the analysis for a master's level project on Data and Text Mining, focusing on stopword behavior within a categorized Vietnamese news corpus.



### 1. Dataset and Models

* **Corpus:** A balanced subset of 16,000 documents (2,000 documents x 8 categories) derived from the [Vietnamese Text Classification Dataset](https://www.kaggle.com/datasets/phamtuyet/text-classification) on Kaggle.
 
* **Stopword List:** A standard [Vietnamese Stopwords](https://github.com/stopwords/vietnamese-stopwords) list from GitHub, used as the baseline for comparison.
 
* **Embedding Models:**
    * Word2Vec ([GitHub](https://github.com/datquocnguyen/PhoW2V?tab=readme-ov-file))
    * FastText ([GitHub](https://github.com/datquocnguyen/PhoW2V?tab=readme-ov-file))
    * PhoBERT ([vinai/phobert-base](https://huggingface.co/vinai/phobert-base-v2))

### 2. Installation

This project was developed in a Jupyter environment. Key Python libraries can be installed via `pip`:

```bash
# Core data handling and ML
pip install pandas numpy scikit-learn

# NLP and Embeddings
pip install underthesea gensim torch transformers

# Data Access and Utilities
pip install kagglehub datasets tqdm
pip install googletrans==4.0.0-rc1

# Visualization
pip install matplotlib seaborn wordcloud
```


Additionally, a font file [Noto Sans by Google](https://fonts.google.com/noto/specimen/Noto+Sans) supporting Vietnamese (e.g., NotoSans-Regular.ttf) is required in the static/ directory for wordcloud to render correctly.

### 3. Project Workflow
The analysis is divided into two main notebooks, designed to be run in sequence. All intermediate results are cached as .pkl files in the pkl_folder to prevent re-computation. All plots are saved in the `plot`/ `folder`.

--- 

#### **Notebook 1: analysis (1- 4) finalize.ipynb**
This notebook handles data preparation and the TF-IDF-based analysis.

    Data Loading: Loads the full news dataset and the stopword list.

    Subsetting: Creates the balanced 16,000-document (2k * 8 categories) vn_df DataFrame.

    Task 1: Analyzes stopword proportion by category (bar plot, tables, and stopword word clouds).

    Task 2: Calculates TF-IDF statistics (avg, std, min, max) for each stopword across categories.

    Task 3: Identifies the Top 100 stopwords based on the lowest average TF-IDF scores and checks overlap with the default list.

    Task 4: Identifies stopwords with a zero TF-IDF score in >= 50% of categories.

#### **Notebook 2: analysis (5-11) finalize.ipynb**
This notebook implements the **Armano et al.** method and conducts the `statistical` vs. `semantic` clustering analysis.

    Task 6: Implements the Armano et al. (φ/δ) metrics using the phi = sensitivity - specificity and delta = sensitivity - fallout formulas. Ranks all words and calculates Precision@N against the default stopword list.

    Task 7: Visualizes the Top 50 Armano stopwords in the φ/δ space.

    Task 8: Applies K-Means clustering to the Top 30 Armano stopwords based on their statistical (φ/δ) features.

    Task 9: Applies K-Means clustering to the default NLTK stopwords based on their statistical (φ/δ) features.

    Task 10: Applies K-Means clustering to both lists (Armano & Default) based on semantic (Word2Vec) features and compares the resulting clusters to Tasks 8 & 9.

    Task 11: Repeats Task 10 using FastText and PhoBERT (as a GloVe replacement) to compare different embedding models.

### 4. How to Run

- Create three folders in the root directory: `pkl_folder`, `plot`, and `static`.

- Download a Vietnamese-compatible font ([Noto Sans by Google](https://fonts.google.com/noto/specimen/Noto+Sans)) and place it in the `font` folder. Update the `font_path` variable in the Task 1 notebook if necessary.

- Ensure your kaggle.json API key is set up for kagglehub to download the datasets.

- Run the notebook analysis (1- 4) finalize.ipynb from top to bottom.

- Run the notebook analysis (5-11) finalize.ipynb from top to bottom.

```
To **re-run** a long calculation (e.g., Task 1, 2, 4,..), delete the corresponding .pkl file (e.g., `task1.pkl`) from the pkl_folder and re-run that cell. 
```

## `Roope Tukiainen` - Analysis of Stopword Behavior in NLTK Reuters and NPS Chat corpuses.

### 1. Dataset and Models

* **NLTK Reuters:** NLTK Reuters-21578 Distribution 1.0. Unbalanced dataset with 90 categories, 10 788 unique documents, and 2 505 678 tokensin total.

* **NLTK NPS Chat:** NLTK NPS Chat Release 1.0. Unbalanced dataset with 15 categories, 10 567 unique posts, and 47 155 tokens in total.
 
* **Embedding Models:**
    * Word2Vec  (gensim.downloader.load("word2vec-google-news-300"))
    * FastText  (gensim.downloader.load("fasttext-wiki-news-subwords-300"))
    * Glove     (gensimdownloader.load("glove-twitter-100"))

### 2. Installation

This project was developed in python 3.10.18 with libraries:
nltk: Used with tokenization, preprocessing, and as a source for corpuses Reuters and NPS Chat.
scikit-learn: K-means algorithm for clustering and t-SNE for dimensionality reduction.
gensim: Used for embedding models and as their downloader.
numpy: Arithematic
pandas: Data manipulation in tabular form.
seaborn and matplotlib: Used for graphical representations of the data.


```bash
# All dependencies can be installed from roope/requirements.txt
pip install -r requirements.txt

# Alternative
pip install nltk pandas numpy scikit-learn gensim matplotlib seaborn
```

### 3. Project Workflow
The code (calculations.py) does all necessary calculations for the analysis and saves the results in the current directory by making necessary files and folders. Some graphical representations are also made and saved into .png files. 
The code has been split into different steps and it will check whether the necessary steps have been completed by checking the current directory for the folders and files it should have created. If those files or directories are not found it will start to recompute everything.

First time downloading all necessary models and corpuses may take time. Step 3 is relative slow and step 6 is extremely long (3 hours on I7-9700k processor). Other steps are rather quick.

### 4. How to Run

- Create a folder in which you will run the calculations.py script and the results will be saved there.

- python calculations.py, should take care of everything and you only need to wait for it complete. It will print of its progress to the console at times, but doesn't estimate how long it will take.


## **References**

### Core Methodology & Supporting Literature

* **Armano, G. et al. (n.d.). "Stopwords identification by means characteristics and discriminants analysis."**
    * The foundational paper for this project's Task 6, providing the φ (characteristic) and δ (discriminant) metrics.


### Datasets, Models & Toolkits

* **Pham, T. (2022). *Vietnamese Text Classification Dataset*. Kaggle.**
    * (Available at: https://www.kaggle.com/datasets/phamtuyet/text-classification)

* **stopwords-iso (n.d.). *Vietnamese Stopwords*. GitHub.**
    * (Available at: https://github.com/stopwords/vietnamese-stopwords)

* **L3VIEVIL (2021). *Vietnamese Stopwords*. Kaggle.**
    * (Alternative source: https://www.kaggle.com/datasets/linhlpv/vietnamese-stopwords)

* **Nguyen, D. Q. (2020). *PhoW2V: Pre-trained Word2Vec & FastText models for Vietnamese*. GitHub.**
    * (Available at: https://github.com/datquocnguyen/PhoW2V)

* **VinAI (2021). *PhoBERT-base (version 2)*. Hugging Face Model Hub.**
    * (Available at: https://huggingface.co/vinai/phobert-base-v2)

* **Vu, A. et al. (2019). *underthesea: Vietnamese NLP toolkit*.**
    * (Available at: https://github.com/undertheseanlp/underthesea)

* **Google (n.d.). *Noto Sans Font Family*. Google Fonts.**
    * (Available at: https://fonts.google.com/noto/specimen/Noto+Sans)