import pytest

from nltk.corpus.reader import CorpusReader


@pytest.fixture(autouse=True)
def mock_plot(mocker):
    """ Disable matplotlib plotting in test code """

    try:
        import matplotlib.pyplot as plt
        mocker.patch.object(plt, 'gca')
        mocker.patch.object(plt, 'show')
    except ImportError:
        pass


@pytest.fixture(scope="session", autouse=True)
def teardown_loaded_corpora(request):
    """
    After each test session ends (either doctest or unit test),
    unload any loaded corpora
    """
    import nltk.corpus

    for name in dir(nltk.corpus):
        obj = getattr(nltk.corpus, name, None)
        if isinstance(obj, CorpusReader) and hasattr(obj, "_unload"):
            obj._unload()
