# Natural Language Toolkit: WordNet stemmer interface
#
# Copyright (C) 2001-2024 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

from typing import List

from nltk.corpus import wordnet as wn


class WordNetLemmatizer:
    """
    WordNet Lemmatizer

    Lemmatize using WordNet's built-in morphy function.
    Returns the input word unchanged if it cannot be found in WordNet.
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
        lemmas = wn._morphy(word, pos)
        return min(lemmas, key=len) if lemmas else word

    def lemmatize_text(self, text: str) -> List[str]:
        """
        Tokenize input text, estimate the part-of-speech tag of each word,
        and return a list of lemmas and pos tag.

        Returns each input word unchanged when it cannot be found in WordNet.

        :param text: The input text to lemmatize.
        :type text: str
        :return: A list with the estimated lemma and pos tag of each `word` in the input text.

            >>> from nltk.stem import WordNetLemmatizer
            >>> wntl = WordNetLemmatizer().lemmatize_text
            >>> print(wntl('Proverbs are short sentences drawn from long experience.'))
            ['Proverbs', 'be', 'short', 'sentence', 'draw', 'from', 'long', 'experience', '.']
            >>> print(wntl('proverbs are short sentences drawn from long experience.'))
            ['proverb', 'be', 'short', 'sentence', 'draw', 'from', 'long', 'experience', '.']
        """
        from nltk.tag import pos_tag
        from nltk.tokenize import word_tokenize

        return [
            # Lemmatize each POS-tagged word:
            (self.lemmatize(word, self.tag2pos(tag)), tag)
            # Tokenize the input text and POS tag each word:
            for word, tag in pos_tag(word_tokenize(text))
        ]

    @staticmethod
    def tag2pos(tag):
        return {"N": "n", "V": "v", "J": "a", "R": "r"}.get(tag[0], "n")

    def __repr__(self):
        return "<WordNetLemmatizer>"
