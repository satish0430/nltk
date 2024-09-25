# Natural Language Toolkit (NLTK)
[![PyPI](https://img.shields.io/pypi/v/nltk.svg)](https://pypi.python.org/pypi/nltk)
![CI](https://github.com/nltk/nltk/actions/workflows/ci.yaml/badge.svg?branch=develop)

NLTK -- the Natural Language Toolkit -- is a suite of open source Python
modules, data sets, and tutorials supporting research and development in Natural
Language Processing. NLTK requires Python version 3.8, 3.9, 3.10, 3.11 or 3.12.

For documentation, please visit [nltk.org](https://www.nltk.org/).

+------------------------------------------------------------------------+
   |Table of Contents                                                       |
   |------------------------------------------------------------------------|
   |  * Intalling Data                                                      |
   |  * Using NLTK POS Tagging                                                        |
   |  * Contributing                                                        |
   |  * Donate                                                              |
   |  * Citing                                                              |
   |  * Copyright                                                           |
   |  * Redistibuting                                                       |
   +------------------------------------------------------------------------+

     ----------------------------------------------------------------------

## Installing Data
NLTK contains many corpora, toy grammars, trained models, etc. To use these run the following python code: 
```
import nltk
nltk.download()
```
This will open up a new window where you can then choose from a comprehensive list of all the data to download. The location of where the data is downloaded can be chnaged through FIle > Change Downlaod Directory.

If you already know the name of the dataset you want to download you can instead run to download it directly
'''
nltk.download('sinica_treebank')
'''
Once you have downloaded data, you can use it by running the import command using the format from nltk import 'name of data' for example if you downloaded the sinica treebank corpus
'''
from nltk.corpus import sinica_treebank
'''
## Using NLTK
One of the features of NLTK is tagging of words. Many of the corpora contained within NLTK have POS tags. To access these tags you first have to download and then import the corpora. Following that you can use the tagged_words() function get a list of key, value pairs of all the tagged words. If you want the tagset to be the universal tagset provide the argument tagset='universal.
'''
nltk.corpus.brown.tagged_words()
nltk.corpus.brown.tagged_words(tagset='universal')
'''

Not all corpora employ the same set of tags so using the univerisal tagset argument can be helpful to get a simpler tagset to work with.

Tagged corpora for several other languages are available with NLTK, including Chinese, Hindi, Portuguese, Spanish, Dutch and Catalan.

NLTK also comes with a pos_tagger that can be used to tag datasets without preexisting tags.
'''
import nltk
text = word_tokenize("Tag this sentence")
nltk.pos_tag(text)
'''
## Contributing

Do you want to contribute to NLTK development? Great!
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

See also [how to contribute to NLTK](https://www.nltk.org/contribute.html).

## Donate

Have you found the toolkit helpful?  Please support NLTK development by donating
to the project via PayPal, using the link on the NLTK homepage.


## Citing

If you publish work that uses NLTK, please cite the NLTK book, as follows:

    Bird, Steven, Edward Loper and Ewan Klein (2009).
    Natural Language Processing with Python.  O'Reilly Media Inc.


## Copyright

Copyright (C) 2001-2024 NLTK Project

For license information, see [LICENSE.txt](LICENSE.txt).

[AUTHORS.md](AUTHORS.md) contains a list of everyone who has contributed to NLTK.


### Redistributing

- NLTK source code is distributed under the Apache 2.0 License.
- NLTK documentation is distributed under the Creative Commons
  Attribution-Noncommercial-No Derivative Works 3.0 United States license.
- NLTK corpora are provided under the terms given in the README file for each
  corpus; all are redistributable and available for non-commercial use.
- NLTK may be freely redistributed, subject to the provisions of these licenses.
