# -*- coding: utf-8 -*-
# Natural Language Toolkit: Distance Metrics
#
# Copyright (C) 2001-2018 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
#         Tom Lippincott <tom@cs.columbia.edu>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
#

"""
Distance Metrics.

Compute the distance between two items (usually strings).
As metrics, they must satisfy the following three requirements:

1. d(a, a) = 0
2. d(a, b) >= 0
3. d(a, c) <= d(a, b) + d(b, c)
"""

from __future__ import print_function
from __future__ import division

import warnings

def _edit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i           # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j           # row 0: 0,1,2,3,4,...
    return lev


def _edit_dist_step(lev, i, j, s1, s2,
                    substitution_cost=1,
                    transpositions=False):
    c1 = s1[i - 1]
    c2 = s2[j - 1]

    # skipping a character in s1
    a = lev[i - 1][j] + 1
    # skipping a character in s2
    b = lev[i][j - 1] + 1
    # substitution
    c = lev[i - 1][j - 1] + (substitution_cost if c1 != c2 else 0)

    # transposition
    d = c + 1  # never picked by default
    if transpositions and i > 1 and j > 1:
        if s1[i - 2] == c2 and s2[j - 2] == c1:
            d = lev[i - 2][j - 2] + 1

    # pick the cheapest
    lev[i][j] = min(a, b, c, d)


def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    """
    Calculate the Levenshtein edit-distance between two strings.
    The edit distance is the number of characters that need to be
    substituted, inserted, or deleted, to transform s1 into s2.  For
    example, transforming "rain" to "shine" requires three steps,
    consisting of two substitutions and one insertion:
    "rain" -> "sain" -> "shin" -> "shine".  These operations could have
    been done in other orders, but at least three steps are needed.

    Allows specifying the cost of substitution edits (e.g., "a" -> "b"),
    because sometimes it makes sense to assign greater penalties to
    substitutions.

    This also optionally allows transposition edits (e.g., "ab" -> "ba"),
    though this is disabled by default.

    :param s1, s2: The strings to be analysed
    :param transpositions: Whether to allow transposition edits
    :type s1: str
    :type s2: str
    :type substitution_cost: int
    :type transpositions: bool
    :rtype int
    """
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _edit_dist_step(lev, i + 1, j + 1, s1, s2,
                            substitution_cost=substitution_cost,
                            transpositions=transpositions)
    return lev[len1][len2]


def binary_distance(label1, label2):
    """Simple equality test.

    0.0 if the labels are identical, 1.0 if they are different.

    >>> from nltk.metrics import binary_distance
    >>> binary_distance(1,1)
    0.0

    >>> binary_distance(1,3)
    1.0
    """

    return 0.0 if label1 == label2 else 1.0


def jaccard_distance(label1, label2):
    """Distance metric comparing set-similarity.

    """
    return (len(label1.union(label2)) -
            len(label1.intersection(label2)))/len(label1.union(label2))


def masi_distance(label1, label2):
    """Distance metric that takes into account partial agreement when multiple
    labels are assigned.

    >>> from nltk.metrics import masi_distance
    >>> masi_distance(set([1, 2]), set([1, 2, 3, 4]))
    0.665

    Passonneau 2006, Measuring Agreement on Set-Valued Items (MASI)
    for Semantic and Pragmatic Annotation.
    """

    len_intersection = len(label1.intersection(label2))
    len_union = len(label1.union(label2))
    len_label1 = len(label1)
    len_label2 = len(label2)
    if len_label1 == len_label2 and len_label1 == len_intersection:
        m = 1
    elif len_intersection == min(len_label1, len_label2):
        m = 0.67
    elif len_intersection > 0:
        m = 0.33
    else:
        m = 0

    return 1 - len_intersection / len_union * m


def interval_distance(label1, label2):
    """Krippendorff's interval distance metric

    >>> from nltk.metrics import interval_distance
    >>> interval_distance(1,10)
    81

    Krippendorff 1980, Content Analysis: An Introduction to its Methodology
    """

    try:
        return pow(label1 - label2, 2)
#        return pow(list(label1)[0]-list(label2)[0],2)
    except:
        print("non-numeric labels not supported with interval distance")


def presence(label):
    """Higher-order function to test presence of a given label
    """

    return lambda x, y: 1.0 * ((label in x) == (label in y))


def fractional_presence(label):
    return lambda x, y:\
        abs(((1.0 / len(x)) - (1.0 / len(y)))) * (label in x and label in y) \
        or 0.0 * (label not in x and label not in y) \
        or abs((1.0 / len(x))) * (label in x and label not in y) \
        or ((1.0 / len(y))) * (label not in x and label in y)


def custom_distance(file):
    data = {}
    with open(file, 'r') as infile:
        for l in infile:
            labelA, labelB, dist = l.strip().split("\t")
            labelA = frozenset([labelA])
            labelB = frozenset([labelB])
            data[frozenset([labelA, labelB])] = float(dist)
    return lambda x, y: data[frozenset([x, y])]


def jaro_similarity(s1, s2):
    """
   Computes the Jaro similarity between 2 sequences from:

        Matthew A. Jaro (1989). Advances in record linkage methodology
        as applied to the 1985 census of Tampa Florida. Journal of the
        American Statistical Association. 84 (406): 414-20.

    The Jaro distance between is the min no. of single-character transpositions
    required to change one word into another. The Jaro similarity formula from
    https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance :

        jaro_sim = 0 if m = 0 else 1/3 * (m/|s_1| + m/s_2 + (m-t)/m)

    where:
        - |s_i| is the length of string s_i
        - m is the no. of matching characters
        - t is the half no. of possible transpositions.

    """
    # First, store the length of the strings
    # because they will be re-used several times.
    len_s1, len_s2 = len(s1), len(s2)

    # The upper bound of the distance for being a matched character.
    match_bound = max(len_s1, len_s2) // 2 - 1

    # Initialize the counts for matches and transpositions.
    matches = 0  # no.of matched characters in s1 and s2
    transpositions = 0  # no. of transpositions between s1 and s2
    flagged_1 = []  # positions in s1 which are matches to some character in s2
    flagged_2 = []  # positions in s2 which are matches to some character in s1

    # Iterate through sequences, check for matches and compute transpositions.
    for i in range(len_s1):     # Iterate through each character.
        upperbound = min(i+match_bound, len_s2-1)
        lowerbound = max(0, i-match_bound)
        for j in range(lowerbound, upperbound+1):
            if s1[i] == s2[j] and j not in flagged_2:
                matches += 1
                flagged_1.append(i)
                flagged_2.append(j)
                break
    flagged_2.sort()
    for i, j in zip(flagged_1, flagged_2):
        if s1[i] != s2[j]:
            transpositions += 1

    if matches == 0:
        return 0
    else:
        return 1/3 * (matches/len_s1 +
                      matches/len_s2 +
                      (matches-transpositions//2)/matches
                      )


def jaro_winkler_similarity(s1, s2, p=0.1, max_l=4):
    """
    The Jaro Winkler distance is an extension of the Jaro similarity in:

        William E. Winkler. 1990. String Comparator Metrics and Enhanced
        Decision Rules in the Fellegi-Sunter Model of Record Linkage.
        Proceedings of the Section on Survey Research Methods.
        American Statistical Association: 354-359.
    such that:

        jaro_winkler_sim = jaro_sim + ( l * p * (1 - jaro_sim) )

    where,

        - jaro_sim is the output from the Jaro Similarity,
        see jaro_similarity()
        - l is the length of common prefix at the start of the string
            - this implementation provides an upperbound for the l value
              to keep the prefixes.A common value of this upperbound is 4.
        - p is the constant scaling factor to overweigh common prefixes.
          The Jaro-Winkler similarity will fall within the [0, 1] bound,
          given that max(p)<=0.25 , default is p=0.1 in Winkler (1990)

    
    Test using outputs from https://www.census.gov/srd/papers/pdf/rr93-8.pdf  
    from "Table 5 Comparison of String Comparators Rescaled between 0 and 1"
    
    >>> winkler_examples = [("billy", "billy"), ("billy", "bill"), ("billy", "blily"), 
    ... ("massie", "massey"), ("yvette", "yevett"), ("billy", "bolly"), ("dwayne", "duane"), 
    ... ("dixon", "dickson"), ("billy", "susan")]
    
    >>> winkler_scores = [1.000, 0.967, 0.947, 0.944, 0.911, 0.893, 0.858, 0.853, 0.000]
    >>> jaro_scores =    [1.000, 0.933, 0.933, 0.889, 0.889, 0.867, 0.822, 0.790, 0.000]

        # One way to match the values on the Winkler's paper is to provide a different 
    # p scaling factor for different pairs of strings, e.g. 
    >>> p_factors = [0.1, 0.125, 0.20, 0.125, 0.20, 0.20, 0.20, 0.15, 0.1]
    
    >>> for (s1, s2), jscore, wscore, p in zip(winkler_examples, jaro_scores, winkler_scores, p_factors):
    ...     assert round(jaro_similarity(s1, s2), 3) == jscore
    ...     assert round(jaro_winkler_similarity(s1, s2, p=p), 3) == wscore

    
    Test using outputs from https://www.census.gov/srd/papers/pdf/rr94-5.pdf from 
    "Table 2.1. Comparison of String Comparators Using Last Names, First Names, and Street Names"
    
    >>> winkler_examples = [('SHACKLEFORD', 'SHACKELFORD'), ('DUNNINGHAM', 'CUNNIGHAM'), 
    ... ('NICHLESON', 'NICHULSON'), ('JONES', 'JOHNSON'), ('MASSEY', 'MASSIE'), 
    ... ('ABROMS', 'ABRAMS'), ('HARDIN', 'MARTINEZ'), ('ITMAN', 'SMITH'), 
    ... ('JERALDINE', 'GERALDINE'), ('MARHTA', 'MARTHA'), ('MICHELLE', 'MICHAEL'), 
    ... ('JULIES', 'JULIUS'), ('TANYA', 'TONYA'), ('DWAYNE', 'DUANE'), ('SEAN', 'SUSAN'), 
    ... ('JON', 'JOHN'), ('JON', 'JAN'), ('BROOKHAVEN', 'BRROKHAVEN'), 
    ... ('BROOK HALLOW', 'BROOK HLLW'), ('DECATUR', 'DECATIR'), ('FITZRUREITER', 'FITZENREITER'), 
    ... ('HIGBEE', 'HIGHEE'), ('HIGBEE', 'HIGVEE'), ('LACURA', 'LOCURA'), ('IOWA', 'IONA'), ('1ST', 'IST')]
    
    >>> jaro_scores =   [0.970, 0.896, 0.926, 0.790, 0.889, 0.889, 0.722, 0.467, 0.926, 
    ... 0.944, 0.869, 0.889, 0.867, 0.822, 0.783, 0.917, 0.000, 0.933, 0.944, 0.905, 
    ... 0.856, 0.889, 0.889, 0.889, 0.833, 0.000]
    
    >>> winkler_scores = [0.982, 0.896, 0.956, 0.832, 0.944, 0.922, 0.722, 0.467, 0.926, 
    ... 0.961, 0.921, 0.933, 0.880, 0.858, 0.805, 0.933, 0.000, 0.947, 0.967, 0.943, 
    ... 0.913, 0.922, 0.922, 0.900, 0.867, 0.000]

        # One way to match the values on the Winkler's paper is to provide a different 
    # p scaling factor for different pairs of strings, e.g. 
    >>> p_factors = [0.1, 0.1, 0.1, 0.1, 0.125, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.20, 
    ... 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
  
    
    >>> for (s1, s2), jscore, wscore, p in zip(winkler_examples, jaro_scores, winkler_scores, p_factors):
    ...     if (s1, s2) in [('JON', 'JAN'), ('1ST', 'IST')]: 
    ...         continue  # Skip bad examples from the paper.
    ...     assert round(jaro_similarity(s1, s2), 3) == jscore
    ...     assert round(jaro_winkler_similarity(s1, s2, p=p), 3) == wscore

    

    This test-case proves that the output of Jaro-Winkler similarity depends on 
    the product  l * p and not on the product max_l * p. Here the product max_l * p > 1
    however the product l * p <= 1
    
    >>> round(jaro_winkler_similarity('TANYA', 'TONYA', p=0.1, max_l=100), 3)
    0.88


    """
    # To ensure that the output of the Jaro-Winkler's similarity 
    # falls between [0,1], the product of l * p needs to be 
    # also fall between [0,1].
    if not 0 <= max_l * p <= 1:
        warnings.warn(str("The product  `max_l * p` might not fall between [0,1]."
              "Jaro-Winkler similarity might not be between 0 and 1.")
             )


    # Compute the Jaro similarity
    jaro_sim = jaro_similarity(s1, s2)

    # Initialize the upper bound for the no. of prefixes.
    # if user did not pre-define the upperbound, 
    # use shorter length between s1 and s2

    # Compute the prefix matches.
    l = 0
    # zip() will automatically loop until the end of shorter string.
    for s1_i, s2_i in zip(s1, s2): 
        if s1_i == s2_i:
            l += 1
        else:
            break
        if l == max_l:
            break
    # Return the similarity value as described in docstring.
    return jaro_sim + (l * p * (1 - jaro_sim))


def levenshtein_distance(s1, s2):
    """
    Calculates the distance between 2 strings as a the number of single-character
    insertions, deletions, or substitutions that it takes to make
    the two strings equivalent.

    Source(s):
        (1) Levenshtein, Vladimir I. (February 1966).
            "Binary codes capable of correcting deletions, insertions, and reversals".
            Soviet Physics Doklady. 10 (8): 707–710.

        (2) https://www.codeproject.com/Articles/13525/Fast-memory-efficient-Levenshtein-algorithm

    :param s1, s2: The strings to be analysed
    :type s1: str
    :type s2: str
    :rtype int
    """
    n1, n2 = len(s1), len(s2)
    mx = max([n1, n2])
    if min([n1, n2]) == 0:
        return mx
    else:
        # must initialize both vectors to the max size so that the longer/shorter string
        # are interchangeable

        v1, v2 = list(range(mx + 1)), list(range(mx + 1))
        for j in v1[1:n1]:
            for i in v2[1:n2]:
                if s1[i] == s2[j]:
                    cost = 0
                else:
                    cost = 1
                # __setitem__() is done in O(1) time.
                v2.__setitem__(j, min(v2[j-1] + 1, v1[j] + 1, v1[j-1] + cost))

    # pop() is done in O(1) as it recalls only the last element.
    return v2.pop()


def damerau_levenshtein_distance(s1, s2):
    """
    Like the Levenshtein Distance, this metric calculates the
    distance between 2 strings as a the number of single-character
    insertions, deletions, or substitutions that it takes to make
    the strings equivalent. However, in this case, transpositions
    are added into the list of available transformation operations.

    Source(s):

        (1) https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance

    Note(s):
        The recursive nature of the function makes it less efficient
        than the one used for levenshtein distance (above).

    :param s1, s2: The strings to be analysed
    :type s1: str
    :type s2: str
    :rtype int

    """

    n1, n2 = len(s1), len(s2)
    if min([len(s1), len(s2)]) == 0:
        return max([len(s1), len(s2)])
    else:
        if s1[len(s1)-1] == s2[len(s2)-1]:
            cost = 0
        else:
            cost = 1

        dist = min(
            [damerau_levenshtein_distance(s1=s1[:len(s1) - 1],
                                          s2=s2) + 1,
             damerau_levenshtein_distance(s1=s1,
                                          s2=s2[:len(s2) - 1]) + 1,
             damerau_levenshtein_distance(s1=s1[:len(s1) - 1],
                                          s2=s2[:len(s2) - 1]) + cost])

        if min([n1, n2]) >= 2 and s1[n1] == s2[n2-1] and s1[n1-1] == s2[n2-1]:
            return min([dist, damerau_levenshtein_distance(s1=s1[:len(s1)-2],
                                                           s2=s2[:len(s2)-2]) + cost])
        else:
            return dist


def hamming_distance(s1, s2):
    """
    The Hamming distance measures the distance between two equal length strings
    and it is determined by counting the positions where the two strings have different
    characters.

    Source(s):

        (1) https://en.wikipedia.org/wiki/Hamming_distance

    :param s1, s2: The strings to be analysed
    :type s1: str
    :type s2: str
    :rtype int

    """
    assert len(s1) == len(s2), "Strings must be the same length."
    return sum(x[0] != x[1] for x in zip(s1, s2))


def lee_distance(s1, s2):
    """
    Lee Distance metric calculates the distance between two strings by re-encoding them
    using a q-ary alphabet of all the unique characters present in one or both strings.
    Then using the q-ary alphabet index values provides the sum of

        min(| string1_i - string2_i | , alphabet_size - | string1_i - string2_i |)

    for each value i in the length of string1 or string2 (these are interchangeable because
    both strings must be of equivalent length).

    Source(s):

        (1) https://en.wikipedia.org/wiki/Lee_distance

    :param s1, s2: The strings to be analysed
    :type s1: str
    :type s2: str
    :rtype int

    """
    assert len(s1) == len(s2), "Strings must be the same length."

    alphabet = set()
    n1, n2 = len(s1), len(s2)
    sx = s1 + s2
    for i in range(n1+n2):
        alphabet.add(sx[i])

    q = len(alphabet)

    if q >= 2:
        s1_enc = _lee_string_encoder(s1, alphabet=list(alphabet))
        s2_enc = _lee_string_encoder(s2, alphabet=list(alphabet))
        return sum([min([abs(s1_enc[i]-s2_enc[i]), q-abs(s1_enc[i]-s2_enc[i])]) for i in range(len(s1))])
    else:
        raise ValueError("number of distinct strings must be greater than 1")


def _lee_string_encoder(string, alphabet):
    for i in range(len(alphabet)):
        for j in range(len(string)):
            if alphabet[i] == string[j]:
                string[j] = i
    return string


def demo():
    string_distance_examples = [("rain", "shine"), ("abcdef", "acbdef"),
                                ("language", "lnaguaeg"), ("language",
                                "lnaugage"), ("language", "lngauage")]
    for s1, s2 in string_distance_examples:
        print("Edit distance btwn '%s' and '%s':" % (s1, s2),
              edit_distance(s1, s2))
        print("Edit dist with transpositions btwn '%s' and '%s':" % (s1, s2),
              edit_distance(s1, s2, transpositions=True))
        print("Jaro similarity btwn '%s' and '%s':" % (s1, s2),
              jaro_similarity(s1, s2))
        print("Jaro-Winkler similarity btwn '%s' and '%s':" % (s1, s2),
              jaro_winkler_similarity(s1, s2))
        print("Jaro-Winkler distance btwn '%s' and '%s':" % (s1, s2),
              1 - jaro_winkler_similarity(s1, s2))
        print("Levenshtein Distance between '%s' and '%s':" % (s1, s2),
              levenshtein_distance(s1, s2))
        print("Damerau-Levenshtein Distance between '%s' and '%s':" % (s1, s2),
              damerau_levenshtein_distance(s1, s2))
        print("Hamming Distance between '%s' and '%s':" % (s1, s2),
              hamming_distance(s1, s2))
        print("Lee Distance between '%s' and '%s':" % (s1, s2),
              lee_distance(s1, s2))

    s1 = set([1, 2, 3, 4])
    s2 = set([3, 4, 5])
    print("s1:", s1)
    print("s2:", s2)
    print("Binary distance:", binary_distance(s1, s2))
    print("Jaccard distance:", jaccard_distance(s1, s2))
    print("MASI distance:", masi_distance(s1, s2))


if __name__ == '__main__':
    demo()
