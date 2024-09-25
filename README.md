# Sentiment Analysis on Tweets

## Project Overview

This project aims to analyze sentiment in tweets to predict signs of depression using natural language processing (NLP) techniques. The idea is that sentiment analysis can be a useful tool in identifying emotional states that could indicate depression, which could help intervene early and potentially prevent severe outcomes.

### Dataset

The dataset used for this project is available on Kaggle: [Sentimental Analysis for Tweets](https://www.kaggle.com/datasets/gargmanas/sentimental-analysis-for-tweets).

### Objective

The main objective is to predict whether a person is showing signs of depression based on the words they use in their social media posts.

## Methodology

### 1. **Data Preprocessing**

To prepare the data for analysis, the following steps were taken:
- **Lemmatization**: Reducing words to their base form.
- **POS Tagging**: Assigning parts of speech to each token.
- **Normalization**: Lowercasing all words and removing accents.
- **Stop Words Removal**: Eliminating common words that do not contribute to sentiment (e.g., "is," "the").
- **Tokenization**: Breaking down the text into individual tokens (words).

### 2. **Feature Engineering**
- **PPMI (Positive Pointwise Mutual Information)**: This technique was used to measure the association between words and depression-related sentiment in the dataset.

### 3. **Modeling**
- **Logistic Regression**: A basic, interpretable machine learning algorithm was used to classify tweets into depressive or non-depressive categories.

### 4. **Results**
- The model achieved an **F1 Score of 93.34**, indicating high accuracy in identifying depressive sentiment in tweets.

## Future Improvements

For further optimization and accuracy, the following improvements could be explored:
- **Consider Capitalized Words**: Words in all caps may indicate anger or resentment, which could be important in detecting depression.
- **Handling Unknown Words**: Implement better techniques to deal with out-of-vocabulary words that the model may not recognize.
- **Static Embeddings**: Use Word2Vec embeddings to capture the meanings of words based on their context.
- **Dynamic Embeddings**: Explore BERT embeddings to capture deeper semantic meanings and context.
- **Neural Networks**: Test models like Recurrent Neural Networks (RNNs) to capture temporal relationships in the data.
- **Fine-Tune Pretrained Models**: Leverage pre-trained language models (e.g., BERT) and fine-tune them on the specific dataset for more nuanced sentiment analysis.

## Conclusion

This project demonstrates how sentiment analysis, when applied to social media data, can serve as a potential tool to predict depressive behavior. By implementing the suggested improvements, the accuracy and effectiveness of the model could be further enhanced.
