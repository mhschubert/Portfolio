# Preprocessing & Tokenization Routine for Twitter Data
### Uses: natural language processing (NLP); Backends: Spacy, Multiprocessing

Implements a preprocessing routine for Twitter text data. The routine is designed for authorship analysis, thus it produces different types of n-gram tokens such as distortion, charcater-based, word-based, lemma-based or functional-based tokens.

To achieve this different modules such as spacy or demoji are used. Moreover, it offers different options of how to parse special unicode characters (replacing them by a token or encasing them, thus keeping them in the data).

The preprocessing is implemented as a multiprocessing-task so that it works effciently on large scale datasets.
The types of preprocessing can be dynamically chosen.

*Note*: Expects a ndjson-file as input and consequently works with minimal effort in collaboration with the Twitter API. Initially written for the PAN-2019-Twitter-celebrity dataset.
