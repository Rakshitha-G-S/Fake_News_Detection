# Fake News Detection

This project is a simple command-line application to detect fake news. It uses a Naive Bayes classifier trained on a dataset of real and fake news articles.

## Description

The script `app.py` loads a dataset of news articles, pre-processes the text data, and trains a Multinomial Naive Bayes model. It then takes a news article as input from the user and predicts whether the news is real or fake.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2.  Install the required Python libraries:
    ```bash
    pip install pandas scikit-learn
    ```

## Usage

1.  Run the application:
    ```bash
    python app.py
    ```
2.  The script will train the model and then prompt you to enter a news article.
3.  Paste the news article and press Enter.
4.  The application will output whether the news is likely to be true or fake.

## Dataset

The dataset consists of two CSV files:

*   `True.csv`: Contains real news articles.
*   `Fake.csv`: Contains fake news articles.

The `app.py` script combines these two datasets and splits them into a training set and a testing set.

## Model

The project uses a Multinomial Naive Bayes (MultinomialNB) classifier from the scikit-learn library. The text data is converted into a matrix of TF-IDF features before being fed into the model. The model achieves an accuracy of over 90% on the test set.


