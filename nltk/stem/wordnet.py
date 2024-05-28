# Natural Language Toolkit: WordNet stemmer interface
#
# Copyright (C) 2001-2024 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

from typing import Iterator, Tuple

from nltk.corpus import wordnet as wn


class WordNetLemmatizer:
    """
    WordNet Lemmatizer

    Methods:
       - lemmatize(word) lemmatizes a word
       - lemmatize_text(text) tokenizes text and lemmatizes each word
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

    def lemmatize_text(self, text: str) -> Iterator[Tuple[str, str]]:
        """Tokenize input text, estimate the Universal Tag of each word,
        lemmatize the result and return an iterator over the (lemma, tag) tuples.
        Returns each input word unchanged, when it cannot be found in WordNet.

        :param text: The input text to lemmatize.
        :type text: str
        :return: an iterator over the estimated (lemma, tag) tuple of each word

        >>> from nltk.stem import WordNetLemmatizer
        >>> wntl = WordNetLemmatizer().lemmatize_text
        >>> print([tup for tup in wntl('Proverbs are short sentences drawn from long experience.')])
        [('Proverbs', 'NOUN'), ('be', 'VERB'), ('short', 'ADJ'), ('sentence', 'NOUN'), ('draw', 'VERB'), ('from', 'ADP'), ('long', 'ADJ'), ('experience', 'NOUN'), ('.', '.')]
        >>> print([tup for tup in wntl('proverbs are short sentences drawn from long experience.')])
        [('proverb', 'NOUN'), ('be', 'VERB'), ('short', 'ADJ'), ('sentence', 'NOUN'), ('draw', 'VERB'), ('from', 'ADP'), ('long', 'ADJ'), ('experience', 'NOUN'), ('.', '.')]
        """
        from nltk.tag import pos_tag
        from nltk.tokenize import word_tokenize

        yield from (
            # Lemmatixe each word using the WordNet Pos corresponding to its
            # Universal tag (or 'n' when WordNet does not cover that Pos):
            (self.lemmatize(word, universal_tag_to_wn_pos(tag) or "n"), tag)
            # Tokenize the input text and POS-tag each word, using Universal Tags:
            for word, tag in pos_tag(word_tokenize(text), tagset="universal")
        )

    def __repr__(self) -> str:
        return "<WordNetLemmatizer>"


def universal_tag_to_wn_pos(tag) -> str:
    """Convert Universal Tag to WordNet Part-of-speech.
    Return None when WordNet does not cover the Pos"""
    return {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}.get(tag, None)
