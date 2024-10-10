import unittest
from sentiment_analysis import classify_text

class TestSentimentAnalysis(unittest.TestCase):
    """
    Unit tests for the classify_text function in the sentiment analysis model.
    These tests validate the classifier's behavior on positive, negative, and empty text inputs.
    """

    def test_positive_text(self):
        """
        Test if a clearly positive sentence is classified as 'pos'.
        """
        result = classify_text("I really love this movie")
        self.assertEqual(result, 'pos', "Should classify as positive")

    def test_negative_text(self):
        """
        Test if a clearly negative sentence is classified as 'neg'.
        """
        result = classify_text("This movie is terrible!")
        self.assertEqual(result, 'neg', "Should classify as negative")

    def test_empty_text(self):
        """
        Test if an empty string returns None, as there is no content to classify.
        """
        result = classify_text("")
        self.assertIsNone(result, "Empty text should return None")

if __name__ == '__main__':
    # This runs all the test cases when the script is executed.
    unittest.main()
