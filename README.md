# NLP-Based Fraud Detection in Social Media Platforms

**A Text Mining and Machine Learning Approach Using TF-IDF, N-Grams, and Naive Bayes**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-See%20Repository-lightgrey)](#license)

## Project Overview

Social media platforms allow users to communicate and share information rapidly, but they can also facilitate the spread of harmful content, misleading information, and deceptive reviews. Automatically identifying such content is an important text classification challenge.

This project explores **Natural Language Processing (NLP), text mining, and machine learning techniques** to classify social media text across three application areas:

- **Cyberbullying detection**
- **Fake review detection**
- **Misinformation detection**

The project investigates how different text feature extraction approaches affect the performance of a Multinomial Naive Bayes classifier.

This work is based on my MSc Information Technology dissertation, completed at the University of the West of Scotland (UWS), London Campus, in December 2023.

## Research Objectives

1. Prepare and preprocess social media text for machine learning.
2. Explore text patterns and word distributions using exploratory data analysis.
3. Extract textual features using Term Frequency–Inverse Document Frequency (TF-IDF).
4. Compare unigram, bigram, and combined unigram–bigram representations.
5. Train and evaluate Multinomial Naive Bayes classifiers.
6. Compare classification performance using accuracy, precision, recall, F1-score, and confusion matrices where applicable.

## Methodology

The project follows a text classification workflow:

1. **Data preparation:** Load the datasets and inspect the text and target labels.
2. **Text preprocessing:** Apply text cleaning and NLP preprocessing techniques, including tokenization and stop-word handling where appropriate.
3. **Exploratory analysis:** Examine word frequencies and visualize prominent terms using word clouds.
4. **Feature extraction:** Represent text using TF-IDF with different n-gram configurations.
5. **Model training:** Train Multinomial Naive Bayes classifiers on the extracted features.
6. **Model evaluation:** Evaluate and compare model predictions using classification metrics and confusion matrices.
7. **Comparative analysis:** Identify which feature configuration performs best for each application area.

### Feature extraction approaches

| Approach | Description |
|---|---|
| Unigram | Represents individual words as features |
| Bigram | Represents consecutive two-word sequences as features |
| Combined unigram + bigram | Uses both individual words and two-word sequences |

## Reported Research Results

The following results were reported in my dissertation.

| Classification task | Best reported feature approach | Accuracy | F1-score |
|---|---|---:|---:|
| Cyberbullying detection | Unigram | 73% | 75.8% |
| Fake review detection | Combined unigram + bigram | 86% | 87% |
| Misinformation detection | Combined unigram + bigram | 81% | 84% |

**Key finding:** The combined unigram–bigram representation achieved the best reported performance for fake review and misinformation detection, while unigram features performed best for cyberbullying detection.

These figures represent the results reported in the dissertation. Reproducing them requires the corresponding datasets, preprocessing steps, train/test split, and model settings.

## Technologies and Tools

- **Programming language:** Python
- **Development environment:** Google Colab and Jupyter Notebook
- **Data processing:** Pandas, NumPy
- **Natural Language Processing:** NLTK
- **Feature extraction:** Scikit-learn TF-IDF and n-gram vectorization
- **Machine learning:** Scikit-learn Multinomial Naive Bayes
- **Visualization:** Matplotlib, WordCloud
- **Evaluation:** Accuracy, precision, recall, F1-score, and confusion matrices

## Repository Structure

```text
fraud-detection-social-media-colab/
│
├── datasets/
│   └── Dataset files, if included
│
├── images/
│   └── Project visualizations and figures
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda_visualization.ipynb
│   ├── 03_tfidf_feature_extraction.ipynb
│   ├── 04_unigram_bigram_naive_bayes.ipynb
│   └── 05_combined_features_comparison.ipynb
│
├── Reports/
│   └── Final_Report.pdf
│
├── .gitignore
├── README.md
└── requirements.txt
```

The directory descriptions are indicative; actual dataset and report filenames may differ.

## Google Colab Notebooks

The notebooks separate the workflow into manageable stages.

| Notebook | Purpose |
|---|---|
| [01 — Data Preprocessing](notebooks/01_data_preprocessing.ipynb) | Prepare and clean text data |
| [02 — EDA and Visualization](notebooks/02_eda_visualization.ipynb) | Explore text patterns and visualize word frequencies |
| [03 — TF-IDF Feature Extraction](notebooks/03_tfidf_feature_extraction.ipynb) | Transform text into numerical features |
| [04 — Unigram and Bigram Naive Bayes](notebooks/04_unigram_bigram_naive_bayes.ipynb) | Train and evaluate classifiers using different n-gram configurations |
| [05 — Combined Features Comparison](notebooks/05_combined_features_comparison.ipynb) | Compare unigram, bigram, and combined feature approaches |

### Open in Google Colab

To launch a notebook in Colab, open it from the repository and select **Open in Colab**, or use the corresponding Colab link below.

- [Open Notebook 1 in Colab](https://colab.research.google.com/github/Santosh2068/fraud-detection-social-media-colab/blob/main/notebooks/01_data_preprocessing.ipynb)
- [Open Notebook 2 in Colab](https://colab.research.google.com/github/Santosh2068/fraud-detection-social-media-colab/blob/main/notebooks/02_eda_visualization.ipynb)
- [Open Notebook 3 in Colab](https://colab.research.google.com/github/Santosh2068/fraud-detection-social-media-colab/blob/main/notebooks/03_tfidf_feature_extraction.ipynb)
- [Open Notebook 4 in Colab](https://colab.research.google.com/github/Santosh2068/fraud-detection-social-media-colab/blob/main/notebooks/04_unigram_bigram_naive_bayes.ipynb)
- [Open Notebook 5 in Colab](https://colab.research.google.com/github/Santosh2068/fraud-detection-social-media-colab/blob/main/notebooks/05_combined_features_comparison.ipynb)

## Installation and Usage

### Option 1: Run in Google Colab

1. Open the desired notebook using one of the links above.
2. Follow the notebook instructions to upload or access the required dataset.
3. Run the cells in sequence.
4. Review the evaluation metrics and visualizations.

The current notebooks expect a CSV dataset with the text column `tweet_text` and target column `label`, unless the notebook specifies otherwise. Adjust the column names to match your dataset when necessary.

### Option 2: Run locally

Clone the repository:

```bash
git clone https://github.com/Santosh2068/fraud-detection-social-media-colab.git
cd fraud-detection-social-media-colab
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebooks directory and execute the notebook cells in sequence.

**Note:** NLTK resources may need to be downloaded when running the notebooks for the first time. Follow any download instructions provided in the notebook.

## Limitations and Future Improvements

Potential extensions to this research include:

- Evaluating additional classifiers, such as Logistic Regression, Support Vector Machines, and ensemble models.
- Investigating transformer-based NLP models.
- Evaluating performance on larger and more diverse datasets.
- Examining class imbalance and the effects of different train/test splits.
- Improving reproducibility through fixed random seeds, documented preprocessing settings, and consistent evaluation procedures.
- Assessing model generalization to unseen social media content.

These are proposed future improvements, not claims about work already completed.

## Dissertation

The project is based on my MSc IT dissertation:

**Natural Language Processing with Text Mining Approach for Fraud Detection in Social Media Platforms**

University of the West of Scotland (UWS), London Campus  
Programme: MSc Information Technology. 
Completion: December 2023

If permitted for public distribution, the full dissertation or an appropriate summary can be accessed from the `Reports/` directory.

## Author

**Santosh Poudel**

MSc Information Technology | Data Analytics | Python | SQL | Machine Learning | Natural Language Processing

GitHub: [@Santosh2068](https://github.com/Santosh2068)

Portfolio: [santoshpoudyal.com.np](https://santoshpoudyal.com.np)

## License

No open-source license has been specified for this repository yet. Until a license is added, reuse and redistribution of the project code should not be assumed to be permitted. Check the rights for the datasets, dissertation, and other third-party materials before publishing them.
