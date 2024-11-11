import nltk
from nltk.corpus import movie_reviews
import random

# Ensure necessary NLTK data is downloaded
nltk.download('movie_reviews')
nltk.download('punkt')

# Prepare the dataset by extracting word lists for each movie review and associating them with their category (pos/neg).
# documents is a list of tuples (list of words, category)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Get the 2000 most frequent words from the movie reviews as features for sentiment classification.
# Convert all words to lowercase for consistency.
all_words = [word.lower() for word in movie_reviews.words()]
all_words_freq = nltk.FreqDist(all_words)  # Frequency distribution of all words
top_words = [word for word, freq in all_words_freq.most_common(2000)]  # Top 2000 words

# Define feature extractor function
def document_features(document):
    """
    Extract features from a document (list of words) to check the presence of top words.

    Args:
    - document (list): A list of words from the document.

    Returns:
    - features (dict): A dictionary where the keys are 'contains(word)' and 
      the values are True if the word is present in the document, False otherwise.
    """
    words = set(document)  # Convert document to a set to optimize lookups
    features = {}
    # For each of the top words, add a feature indicating its presence in the document
    for word in top_words:
        features[f'contains({word})'] = (word in words)
    return features

# Shuffle the documents to ensure random distribution between training and test sets
random.shuffle(documents)

# Prepare the feature sets by applying document_features to each document in the dataset.
# featuresets is a list of tuples (features, category)
featuresets = [(document_features(doc), category) for (doc, category) in documents]

# Split the dataset into training (first 1500 examples) and testing sets (next 500 examples).
train_set = featuresets[:1500]  # Training set: 1500 examples
test_set = featuresets[1500:2000]  # Test set: 500 examples

# Train the Naive Bayes classifier on the training set.
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Function to classify a new text input
def classify_text(text):
    """
    Classify a given text input as either 'pos' or 'neg' using the trained Naive Bayes classifier.

    Args:
    - text (str): The input text to be classified.

    Returns:
    - str: The predicted category ('pos' or 'neg').
           Returns None if the input text is empty or contains only whitespace.
    """
    if not text.strip():  # Check for empty input or whitespace
        return None
    # Tokenize the input text and extract features for classification
    features = document_features(nltk.word_tokenize(text))
    return classifier.classify(features)

# Calculate and print classifier accuracy on the test set
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f"Classifier accuracy: {accuracy:.2f}")