# 🏰 Clash Royale Review Sentiment Analysis

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-green)](https://scikit-learn.org)

## 📖 Project Overview
This project classifies user sentiment for the mobile game **Clash Royale** based on over 12,000 reviews scraped from the Google Play Store. It demonstrates a complete data science workflow, including automated data acquisition, **Lexicon-based semi-automated labeling**, and a comparative study between **Traditional Machine Learning** and **Deep Learning** architectures.



## 🚀 Key Features
* **Automated Data Acquisition**: Custom scraping of +12,000 reviews using the `google-play-scraper` library.
* **Semi-Automated Labeling (Lexicon-based)**: Addressed potential label noise by validating star ratings against text sentiment using the **VADER Lexicon**. Reviews with high contradictions (e.g., 5-star rating with highly negative text) were filtered to ensure high-quality training data.
* **Triple-Scheme Experimentation**:
    * **Scheme 1 (Classic ML)**: Support Vector Machine (SVM) + TF-IDF Vectorization.
    * **Scheme 2 (Deep Learning)**: 1D Convolutional Neural Network (CNN) + Word Embeddings.
    * **Scheme 3 (Deep Learning)**: Hybrid CNN-LSTM Architecture for complex sequence understanding.
* **Performance**: Achieved high reliability with all models scoring above **93% testing accuracy**.

## 📊 Experimental Results
The models were evaluated on a balanced dataset where star ratings were validated and mapped into three classes: **Negative (0)**, **Neutral (1)**, and **Positive (2)**.

| Scheme | Algorithm Type | Algorithm | Feature Extraction | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Scheme 1** | **Traditional ML** | **SVM** | TF-IDF | ~93.21% |
| **Scheme 2** | **Deep Learning** | **CNN 1D** | Word Embedding | **~94.42%** |
| **Scheme 3** | **Deep Learning** | **Hybrid CNN-LSTM** | Word Embedding | ~93.89% |



## 🛠️ Installation & Setup
1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/clash-royale-sentiment.git](https://github.com/your-username/clash-royale-sentiment.git)
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Project Structure
* `scraping_clash_royale.py`: Python script for automated data scraping from Play Store.
* `Clash_Royale_Sentiment.ipynb`: Main notebook containing data preprocessing, Lexicon validation, and model training for all 3 schemes.
* `inference.ipynb`: Specialized notebook for loading saved models and testing on custom text.
* `model_cnn_clash_royale.keras`: The best-performing saved model (CNN 1D).
* `tokenizer.pickle`: Saved tokenizer to ensure text-to-sequence consistency during inference.
* `requirements.txt`: List of required Python libraries.

## 🧪 Usage (Inference)
To run sentiment predictions on new reviews without retraining:
1.  Ensure `model_cnn_clash_royale.keras` and `tokenizer.pickle` are in the project root.
2.  Open and run the `inference.ipynb` notebook.
3.  Input your text into the `predict_sentiment()` function.

```python
# Example Usage
review = "The new updates are amazing, I love the strategy!"
label, confidence = predict_sentiment(review)
print(f"Result: {label} ({confidence:.2f}%)")
```

## 📝 Methodology Note
To improve labeling quality beyond simple star ratings, this project utilizes VADER (Valence Aware Dictionary and sEntiment Reasoner). By filtering out "noisy" data where the star rating and lexicon score were fundamentally opposed (e.g. Catching sarcasm or accidental mis-ratings), the models achieved higher convergence rates and more reliable evaluation metrics.
