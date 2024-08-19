# Natural Language Toolkit: IBM Model 4
#
# Copyright (C) 2001-2024 NLTK Project
# Author: Tah Wei Hoon <hoon.tw@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Translation model that reorders output words based on their type and
distance from other related words in the output sentence.

IBM Model 4 improves the distortion model of Model 3, motivated by the
observation that certain words tend to be re-ordered in a predictable
way relative to one another. For example, <adjective><noun> in English
usually has its order flipped as <noun><adjective> in French.

Model 4 requires words in the source and target vocabularies to be
categorized into classes. This can be linguistically driven, like parts
of speech (adjective, nouns, prepositions, etc). Word classes can also
be obtained by statistical methods. The original IBM Model 4 uses an
information theoretic approach to group words into 50 classes for each
vocabulary.

Terminology
-----------

:Cept:
    A source word with non-zero fertility i.e. aligned to one or more
    target words.
:Tablet:
    The set of target word(s) aligned to a cept.
:Head of cept:
    The first word of the tablet of that cept.
:Center of cept:
    The average position of the words in that cept's tablet. If the
    value is not an integer, the ceiling is taken.
    For example, for a tablet with words in positions 2, 5, 6 in the
    target sentence, the center of the corresponding cept is
    ceil((2 + 5 + 6) / 3) = 5
:Displacement:
    For a head word, defined as (position of head word - position of
    previous cept's center). Can be positive or negative.
    For a non-head word, defined as (position of non-head word -
    position of previous word in the same tablet). Always positive,
    because successive words in a tablet are assumed to appear to the
    right of the previous word.

In contrast to Model 3 which reorders words in a tablet independently of
other words, Model 4 distinguishes between three cases.

1. Words generated by NULL are distributed uniformly.
2. For a head word t, its position is modeled by the probability
   d_head(displacement | word_class_s(s),word_class_t(t)),
   where s is the previous cept, and word_class_s and word_class_t maps
   s and t to a source and target language word class respectively.
3. For a non-head word t, its position is modeled by the probability
   d_non_head(displacement | word_class_t(t))

The EM algorithm used in Model 4 is:

:E step: In the training data, collect counts, weighted by prior
         probabilities.

         - (a) count how many times a source language word is translated
               into a target language word
         - (b) for a particular word class, count how many times a head
               word is located at a particular displacement from the
               previous cept's center
         - (c) for a particular word class, count how many times a
               non-head word is located at a particular displacement from
               the previous target word
         - (d) count how many times a source word is aligned to phi number
               of target words
         - (e) count how many times NULL is aligned to a target word

:M step: Estimate new probabilities based on the counts from the E step

Like Model 3, there are too many possible alignments to consider. Thus,
a hill climbing approach is used to sample good candidates.

Notations
---------

:i: Position in the source sentence
     Valid values are 0 (for NULL), 1, 2, ..., length of source sentence
:j: Position in the target sentence
     Valid values are 1, 2, ..., length of target sentence
:l: Number of words in the source sentence, excluding NULL
:m: Number of words in the target sentence
:s: A word in the source language
:t: A word in the target language
:phi: Fertility, the number of target words produced by a source word
:p1: Probability that a target word produced by a source word is
     accompanied by another target word that is aligned to NULL
:p0: 1 - p1
:dj: Displacement, Δj

References
----------

Philipp Koehn. 2010. Statistical Machine Translation.
Cambridge University Press, New York.

Peter E Brown, Stephen A. Della Pietra, Vincent J. Della Pietra, and
Robert L. Mercer. 1993. The Mathematics of Statistical Machine
Translation: Parameter Estimation. Computational Linguistics, 19 (2),
263-311.
"""

import warnings
from collections import defaultdict
from math import factorial

from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel3
from nltk.translate.ibm_model import Counts, longest_target_sentence_length


class IBMModel4(IBMModel):
    """
    Translation model that reorders output words based on their type and
    their distance from other related words in the output sentence

    >>> bitext = []
    >>> bitext.append(AlignedSent(['klein', 'ist', 'das', 'haus'], ['the', 'house', 'is', 'small']))
    >>> bitext.append(AlignedSent(['das', 'haus', 'war', 'ja', 'groß'], ['the', 'house', 'was', 'big']))
    >>> bitext.append(AlignedSent(['das', 'buch', 'ist', 'ja', 'klein'], ['the', 'book', 'is', 'small']))
    >>> bitext.append(AlignedSent(['ein', 'haus', 'ist', 'klein'], ['a', 'house', 'is', 'small']))
    >>> bitext.append(AlignedSent(['das', 'haus'], ['the', 'house']))
    >>> bitext.append(AlignedSent(['das', 'buch'], ['the', 'book']))
    >>> bitext.append(AlignedSent(['ein', 'buch'], ['a', 'book']))
    >>> bitext.append(AlignedSent(['ich', 'fasse', 'das', 'buch', 'zusammen'], ['i', 'summarize', 'the', 'book']))
    >>> bitext.append(AlignedSent(['fasse', 'zusammen'], ['summarize']))
    >>> src_classes = {'the': 0, 'a': 0, 'small': 1, 'big': 1, 'house': 2, 'book': 2, 'is': 3, 'was': 3, 'i': 4, 'summarize': 5 }
    >>> trg_classes = {'das': 0, 'ein': 0, 'haus': 1, 'buch': 1, 'klein': 2, 'groß': 2, 'ist': 3, 'war': 3, 'ja': 4, 'ich': 5, 'fasse': 6, 'zusammen': 6 }

    >>> ibm4 = IBMModel4(bitext, 5, src_classes, trg_classes)

    >>> print(round(ibm4.translation_table['buch']['book'], 3))
    1.0
    >>> print(round(ibm4.translation_table['das']['book'], 3))
    0.0
    >>> print(round(ibm4.translation_table['ja'][None], 3))
    1.0

    >>> print(round(ibm4.head_distortion_table[1][0][1], 3))
    1.0
    >>> print(round(ibm4.head_distortion_table[2][0][1], 3))
    0.0
    >>> print(round(ibm4.non_head_distortion_table[3][6], 3))
    0.5

    >>> print(round(ibm4.fertility_table[2]['summarize'], 3))
    1.0
    >>> print(round(ibm4.fertility_table[1]['book'], 3))
    1.0

    >>> print(round(ibm4.p1, 3))
    0.033

    >>> test_sentence = bitext[2]
    >>> test_sentence.words
    ['das', 'buch', 'ist', 'ja', 'klein']
    >>> test_sentence.mots
    ['the', 'book', 'is', 'small']
    >>> test_sentence.alignment
    Alignment([(0, 0), (1, 1), (2, 2), (3, None), (4, 3)])

    """

    def __init__(
        self,
        sentence_aligned_corpus,
        iterations,
        source_word_classes,
        target_word_classes,
        probability_tables=None,
    ):
        """
        Train on ``sentence_aligned_corpus`` and create a lexical
        translation model, distortion models, a fertility model, and a
        model for generating NULL-aligned words.

        Translation direction is from ``AlignedSent.mots`` to
        ``AlignedSent.words``.

        :param sentence_aligned_corpus: Sentence-aligned parallel corpus
        :type sentence_aligned_corpus: list(AlignedSent)

        :param iterations: Number of iterations to run training algorithm
        :type iterations: int

        :param source_word_classes: Lookup table that maps a source word
            to its word class, the latter represented by an integer id
        :type source_word_classes: dict[str]: int

        :param target_word_classes: Lookup table that maps a target word
            to its word class, the latter represented by an integer id
        :type target_word_classes: dict[str]: int

        :param probability_tables: Optional. Use this to pass in custom
            probability values. If not specified, probabilities will be
            set to a uniform distribution, or some other sensible value.
            If specified, all the following entries must be present:
            ``translation_table``, ``alignment_table``,
            ``fertility_table``, ``p1``, ``head_distortion_table``,
            ``non_head_distortion_table``. See ``IBMModel`` and
            ``IBMModel4`` for the type and purpose of these tables.
        :type probability_tables: dict[str]: object
        """
        super().__init__(sentence_aligned_corpus)
        self.reset_probabilities()
        self.src_classes = source_word_classes
        self.trg_classes = target_word_classes

        if probability_tables is None:
            # Get probabilities from IBM model 3
            ibm3 = IBMModel3(sentence_aligned_corpus, iterations)
            self.translation_table = ibm3.translation_table
            self.alignment_table = ibm3.alignment_table
            self.fertility_table = ibm3.fertility_table
            self.p1 = ibm3.p1
            self.set_uniform_probabilities(sentence_aligned_corpus)
        else:
            # Set user-defined probabilities
            self.translation_table = probability_tables["translation_table"]
            self.alignment_table = probability_tables["alignment_table"]
            self.fertility_table = probability_tables["fertility_table"]
            self.p1 = probability_tables["p1"]
            self.head_distortion_table = probability_tables["head_distortion_table"]
            self.non_head_distortion_table = probability_tables[
                "non_head_distortion_table"
            ]

        for n in range(0, iterations):
            self.train(sentence_aligned_corpus)

    def reset_probabilities(self):
        super().reset_probabilities()
        self.head_distortion_table = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: self.MIN_PROB))
        )
        """
        dict[int][int][int]: float. Probability(displacement of head
        word | word class of previous cept,target word class).
        Values accessed as ``distortion_table[dj][src_class][trg_class]``.
        """

        self.non_head_distortion_table = defaultdict(
            lambda: defaultdict(lambda: self.MIN_PROB)
        )
        """
        dict[int][int]: float. Probability(displacement of non-head
        word | target word class).
        Values accessed as ``distortion_table[dj][trg_class]``.
        """

    def set_uniform_probabilities(self, sentence_aligned_corpus):
        """
        Set distortion probabilities uniformly to
        1 / cardinality of displacement values
        """
        max_m = longest_target_sentence_length(sentence_aligned_corpus)

        # The maximum displacement is m-1, when a word is in the last
        # position m of the target sentence and the previously placed
        # word is in the first position.
        # Conversely, the minimum displacement is -(m-1).
        # Thus, the displacement range is (m-1) - (-(m-1)). Note that
        # displacement cannot be zero and is not included in the range.
        if max_m <= 1:
            initial_prob = IBMModel.MIN_PROB
        else:
            initial_prob = 1 / (2 * (max_m - 1))
        if initial_prob < IBMModel.MIN_PROB:
            warnings.warn(
                "A target sentence is too long ("
                + str(max_m)
                + " words). Results may be less accurate."
            )

        for dj in range(1, max_m):
            self.head_distortion_table[dj] = defaultdict(
                lambda: defaultdict(lambda: initial_prob)
            )
            self.head_distortion_table[-dj] = defaultdict(
                lambda: defaultdict(lambda: initial_prob)
            )
            self.non_head_distortion_table[dj] = defaultdict(lambda: initial_prob)
            self.non_head_distortion_table[-dj] = defaultdict(lambda: initial_prob)

    def train(self, parallel_corpus):
        counts = Model4Counts()
        for aligned_sentence in parallel_corpus:
            m = len(aligned_sentence.words)

            # Sample the alignment space
            sampled_alignments, best_alignment = self.sample(aligned_sentence)
            # Record the most probable alignment
            aligned_sentence.alignment = Alignment(
                best_alignment.zero_indexed_alignment()
            )

            # E step (a): Compute normalization factors to weigh counts
            total_count = self.prob_of_alignments(sampled_alignments)

            # E step (b): Collect counts
            for alignment_info in sampled_alignments:
                count = self.prob_t_a_given_s(alignment_info)
                normalized_count = count / total_count

                for j in range(1, m + 1):
                    counts.update_lexical_translation(
                        normalized_count, alignment_info, j
                    )
                    counts.update_distortion(
                        normalized_count,
                        alignment_info,
                        j,
                        self.src_classes,
                        self.trg_classes,
                    )

                counts.update_null_generation(normalized_count, alignment_info)
                counts.update_fertility(normalized_count, alignment_info)

        # M step: Update probabilities with maximum likelihood estimates
        # If any probability is less than MIN_PROB, clamp it to MIN_PROB
        existing_alignment_table = self.alignment_table
        self.reset_probabilities()
        self.alignment_table = existing_alignment_table  # don't retrain

        self.maximize_lexical_translation_probabilities(counts)
        self.maximize_distortion_probabilities(counts)
        self.maximize_fertility_probabilities(counts)
        self.maximize_null_generation_probabilities(counts)

    def maximize_distortion_probabilities(self, counts):
        head_d_table = self.head_distortion_table
        for dj, src_classes in counts.head_distortion.items():
            for s_cls, trg_classes in src_classes.items():
                for t_cls in trg_classes:
                    estimate = (
                        counts.head_distortion[dj][s_cls][t_cls]
                        / counts.head_distortion_for_any_dj[s_cls][t_cls]
                    )
                    head_d_table[dj][s_cls][t_cls] = max(estimate, IBMModel.MIN_PROB)

        non_head_d_table = self.non_head_distortion_table
        for dj, trg_classes in counts.non_head_distortion.items():
            for t_cls in trg_classes:
                estimate = (
                    counts.non_head_distortion[dj][t_cls]
                    / counts.non_head_distortion_for_any_dj[t_cls]
                )
                non_head_d_table[dj][t_cls] = max(estimate, IBMModel.MIN_PROB)

    def prob_t_a_given_s(self, alignment_info):
        """
        Probability of target sentence and an alignment given the
        source sentence
        """
        return IBMModel4.model4_prob_t_a_given_s(alignment_info, self)

    @staticmethod  # exposed for Model 5 to use
    def model4_prob_t_a_given_s(alignment_info, ibm_model):
        probability = 1.0
        MIN_PROB = IBMModel.MIN_PROB

        def null_generation_term():
            # Binomial distribution: B(m - null_fertility, p1)
            value = 1.0
            p1 = ibm_model.p1
            p0 = 1 - p1
            null_fertility = alignment_info.fertility_of_i(0)
            m = len(alignment_info.trg_sentence) - 1
            value *= pow(p1, null_fertility) * pow(p0, m - 2 * null_fertility)
            if value < MIN_PROB:
                return MIN_PROB

            # Combination: (m - null_fertility) choose null_fertility
            for i in range(1, null_fertility + 1):
                value *= (m - null_fertility - i + 1) / i
            return value

        def fertility_term():
            value = 1.0
            src_sentence = alignment_info.src_sentence
            for i in range(1, len(src_sentence)):
                fertility = alignment_info.fertility_of_i(i)
                value *= (
                    factorial(fertility)
                    * ibm_model.fertility_table[fertility][src_sentence[i]]
                )
                if value < MIN_PROB:
                    return MIN_PROB
            return value

        def lexical_translation_term(j):
            t = alignment_info.trg_sentence[j]
            i = alignment_info.alignment[j]
            s = alignment_info.src_sentence[i]
            return ibm_model.translation_table[t][s]

        def distortion_term(j):
            t = alignment_info.trg_sentence[j]
            i = alignment_info.alignment[j]
            if i == 0:
                # case 1: t is aligned to NULL
                return 1.0
            if alignment_info.is_head_word(j):
                # case 2: t is the first word of a tablet
                previous_cept = alignment_info.previous_cept(j)
                src_class = None
                if previous_cept is not None:
                    previous_s = alignment_info.src_sentence[previous_cept]
                    src_class = ibm_model.src_classes[previous_s]
                trg_class = ibm_model.trg_classes[t]
                dj = j - alignment_info.center_of_cept(previous_cept)
                return ibm_model.head_distortion_table[dj][src_class][trg_class]

            # case 3: t is a subsequent word of a tablet
            previous_position = alignment_info.previous_in_tablet(j)
            trg_class = ibm_model.trg_classes[t]
            dj = j - previous_position
            return ibm_model.non_head_distortion_table[dj][trg_class]

        # end nested functions

        # Abort computation whenever probability falls below MIN_PROB at
        # any point, since MIN_PROB can be considered as zero
        probability *= null_generation_term()
        if probability < MIN_PROB:
            return MIN_PROB

        probability *= fertility_term()
        if probability < MIN_PROB:
            return MIN_PROB

        for j in range(1, len(alignment_info.trg_sentence)):
            probability *= lexical_translation_term(j)
            if probability < MIN_PROB:
                return MIN_PROB

            probability *= distortion_term(j)
            if probability < MIN_PROB:
                return MIN_PROB

        return probability


class Model4Counts(Counts):
    """
    Data object to store counts of various parameters during training.
    Includes counts for distortion.
    """

    def __init__(self):
        super().__init__()
        self.head_distortion = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )
        self.head_distortion_for_any_dj = defaultdict(lambda: defaultdict(float))
        self.non_head_distortion = defaultdict(lambda: defaultdict(float))
        self.non_head_distortion_for_any_dj = defaultdict(float)

    def update_distortion(self, count, alignment_info, j, src_classes, trg_classes):
        i = alignment_info.alignment[j]
        t = alignment_info.trg_sentence[j]
        if i == 0:
            # case 1: t is aligned to NULL
            pass
        elif alignment_info.is_head_word(j):
            # case 2: t is the first word of a tablet
            previous_cept = alignment_info.previous_cept(j)
            if previous_cept is not None:
                previous_src_word = alignment_info.src_sentence[previous_cept]
                src_class = src_classes[previous_src_word]
            else:
                src_class = None
            trg_class = trg_classes[t]
            dj = j - alignment_info.center_of_cept(previous_cept)
            self.head_distortion[dj][src_class][trg_class] += count
            self.head_distortion_for_any_dj[src_class][trg_class] += count
        else:
            # case 3: t is a subsequent word of a tablet
            previous_j = alignment_info.previous_in_tablet(j)
            trg_class = trg_classes[t]
            dj = j - previous_j
            self.non_head_distortion[dj][trg_class] += count
            self.non_head_distortion_for_any_dj[trg_class] += count
