# Project Description
This project aims to create a text summarization system using both traditional machine learning techniques (TF-IDF and Linear Regression) and a fine-tuned GPT-2 model. The system processes text from sports related sources (news articles and  Reddit posts) and generates concise summaries of the input content. It uses multiple methods for text preprocessing, sentence scoring, and summarization to extract relevant sentences from the articles and posts. The goal is to provide a comprehensive tool that can summarize both structured and unstructured sports related texts.
# Dataset
The primary datasets used in the project are:

• BBC News Dataset: This includes news articles and corresponding summaries, particularly in the "sport" category. The articles are preprocessed by removing special characters and stopwords, and their corresponding summaries are used for training machine learning models.

• Reddit Dataset: Reddit posts from the "sports discussion" subreddit are collected using the PRAW library. These posts include titles, self-text content, and comments, which are preprocessed and used for summarization.

For training the GPT-2 model, the articles and summaries are formatted in a specific way (using a <|startoftext|> and <|endoftext|> token structure), allowing the model to generate summaries after fine-tuning measures have been enacted.
# Your project framework 
The framework consists of the following components:
1. Data Collection & Preprocessing: 
   1. The first step involves collecting text data (news articles and Reddit posts). Preprocessing includes text cleaning, punctuation removal, and stopword elimination.
2. Text Vectorization:
   1. TF-IDF Vectorizer: Converts the text into numerical representations that can be used for modeling.
3. Modeling:
   1. Traditional Summarization (TF-IDF + Cosine Similarity): Sentences in an article are ranked based on the sum of their TF-IDF scores, and the top sentences are selected for the summary.
   2. Fine-Tuned GPT-2 Model: The GPT-2 model is fine-tuned on the news articles and summaries to predict a summary for unseen articles or Reddit posts.
4. Summarization:
   1. The final step generates summaries either by extracting key sentences (using TF-IDF and cosine similarity) or by generating summaries using the GPT-2 model.
   
   ![Flow Chart.drawio.png](Flow%20Chart.drawio.png)
# Results
The project generates concise summaries of articles and Reddit posts.
### TF-IDF Based Summarization
This method identifies the most important sentences based on their TF-IDF scores. It's effective for generating simple, extractive summaries.
### GPT-2 Model
After fine-tuning on the news articles and summaries, the GPT-2 model generates more coherent and contextually relevant summaries. The model can summarize unseen articles or Reddit posts effectively, showing a significant improvement over the traditional method in terms of readability and contextual relevance.
Example Generated Summary (GPT-2): For an article like "Saquon Barkley rushes for 255 yards, sets Eagles record in win over Rams", the model would generate a concise summary, 
focusing on the main events and achievements in the content. It produced a good/fair Gouge score:

```ROUGE Scores: {'rouge1': np.float64(0.6215615793265203), 'rouge2': np.float64(0.5823896609995791), 'rougeL': np.float64(0.39069661554102597), 'rougeLsum': np.float64(0.4795879587686361)}```

# Pre-requisite
The project requires the following software dependencies:

Python 3.7+: The code uses libraries and frameworks compatible with Python 3.7 and above.

Libraries:
```
nltk (version 3.5 or above) for tokenization and stopwords.
scikit-learn (version 0.24 or above) for TF-IDF vectorization and cosine similarity.
numpy (version 1.19 or above) for handling arrays.
pytorch (version 1.7 or above) for model training with GPT-2.
transformers (version 4.5 or above) for working with GPT-2.
torch (2.5.1 + cu118)
datasets (version 1.6 or above) for handling dataset preprocessing.
praw (version 7.0 or above) for Reddit data collection.
evaluate (0.4.3)
rouge_score (0.1.2)
```

Hardware:
```
A machine with a GPU is recommended for fine-tuning the GPT-2 model (since training requires substantial computational power).
```

# How to run 
To run the fine-tuned GPT2, follow these steps:

1. Install Dependencies: First, install the required libraries:

         ```pip install nltk scikit-learn numpy pytorch transformers datasets praw```
2. Download ```gpt2-sports-summary``` folder, which contains the trained model
3. Open ```Gpt-Based Model.ipynb```, run blocks and follow the prompt (It is advised that the input string should be relatively short given we have an input size limit, to achieve a better performance)


4. For summarizing Reddit posts, you can directly call the ```process_reddit_posts()``` function in your main script (either ```Summary Generation-gpt.ipynb``` or ```Summary Generation-TfIdf.ipynb```) after collecting the posts with PRAW.
This approach allows the flexibility of using both traditional extractive summarization (TF-IDF) and advanced abstractive summarization (GPT-2).

# Further Discussion
Along the way of developing and testing models, we faced and are still facing several issues and challenges. We would like to address a few here:
1. Gpt2.0 model tends to suffer from hallucinating, where it generates new information from the given source. Though the idea the generated message conveys is close to the source, it is not ideal;
2. Lack of customization of the tokenizer. We suggest adding a list of known names of athletes and sports-specific terms/slang to the pretrained tokenizer vocabulary to achieve better performance;
3. Insufficient learning material. We propose to use PRAW to continuously scrape content off Reddit which then can be used as training data to better train our model;
4. Insufficient validation method. Since our model is more of a generative model, it is hard to find good data to compare its output against. Also, even though we used Rouge to score our model, human verification would be much better as an evaluation method;
5. Reddit structure/content. During our research, we found out that many posts on Reddit contain only a link or an image, which are very invalid for our training purpose. Also, the structure of Reddit posts and replies are hard to navigate through the raw data.