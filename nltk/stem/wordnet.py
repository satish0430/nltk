# Natural Language Toolkit: WordNet stemmer interface
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

from typing import List

from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


class WordNetLemmatizer:
    """
    WordNet Lemmatizer

    Lemmatize using WordNet's built-in morphy function.
    Returns the input word unchanged if it cannot be found in WordNet.

        >>> from nltk.stem import WordNetLemmatizer
        >>> wnl = WordNetLemmatizer()
        >>> print(wnl.lemmatize('dogs'))
        dog
        >>> print(wnl.lemmatize('churches'))
        church
        >>> print(wnl.lemmatize('aardwolves'))
        aardwolf
        >>> print(wnl.lemmatize('abaci'))
        abacus
        >>> print(wnl.lemmatize('hardrock'))
        hardrock
    """

    def lemmatize(self, word: str, pos: str = "n") -> str:
        """Lemmatize `word` using WordNet's built-in morphy function.
        Returns the input word unchanged if it cannot be found in WordNet.

        :param word: The input word to lemmatize.
        :type word: str
        :param pos: The Part Of Speech tag. Valid options are `"n"` for nouns,
            `"v"` for verbs, `"a"` for adjectives, `"r"` for adverbs and `"s"`
            for satellite adjectives.
        :type pos: str
        :return: The lemma of `word`, for the given `pos`.
        """
        lemmas = wn._morphy(word, pos)
        return min(lemmas, key=len) if lemmas else word

    def __repr__(self):
        return "<WordNetLemmatizer>"


class TextLemmatizer:
    """
    WordNet Text Lemmatizer

    Lemmatize from corpus using word_tokenize, WordNet's built-in morphy function and pos_tag.
    Returns the input word unchanged if it cannot be found in WordNet.

        >>> from nltk.stem import TextLemmatizer
        >>> text_wnl = TextLemmatizer()
        >>> print(text_wnl.lemmatize('Proverbs are short sentences drawn from long experience.'))
        ['Proverbs', 'be', 'short', 'sentence', 'draw', 'from', 'long', 'experience', '.']
        >>> print(text_wnl.auto_lemmatize('proverbs are short sentences drawn from long experience.'))
        ['proverb', 'be', 'short', 'sentence', 'draw', 'from', 'long', 'experience', '.']
    """

    # POS tag dict for matching with WordNet's lemmatize
    pos_word_dict = {
        "VBP": "v",
        "VB": "v",
        "VBG": "v",
        "VBD": "v",
        "VBN": "v",
        "VBZ": "v",
        "JJ": "a",
        "JJR": "a",
        "JJS": "a",
        "RB": "r",
        "RBR": "r",
        "RBS": "r",
        "NN": "n",
        "NNS": "n",
        "NNP": "n",
        "NNPS": "n",
    }

    def lemmatize(self, sentence: str) -> List:
        """Automatically lemmatize `word` with out pos using pos_tag and WordNet's built-in morphy function.
        Returns the input word unchanged if it cannot be found in WordNet.

        :param word: The input word to lemmatize.
        :type word: str
        :return: The list for lemma of `word`, for automatically estimates `pos`.
        """
        # Tokenize the sentence
        words = word_tokenize(sentence)

        # POS tagging
        pos_tags = pos_tag(words)

        # Lemmatize the words
        lemma_list = []
        for i, word in enumerate(words):
            lemma = WordNetLemmatizer().lemmatize(
                word, self.pos_word_dict.get(pos_tags[i][1], "n")
            )  # word.lower() can be used but it is trade-off problem
            lemma_list.append(lemma)

        return lemma_list
