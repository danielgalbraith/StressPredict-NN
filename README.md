# StressPredict-NN
Multi-class softmax classification with uni-directional neural network for predicting sentence stress annotation. Trained on data from US Presidential Inaugural Addresses with gold-standard human annotations. Uses Keras Sequential model in Python 3.

* **Input features**: word index within sentence chunk, lexical stress {yes, no, ambiguous}, number of segments in word, number of syllables in word, number of stresses, part of speech, dependency parse, frequency of word in document, informativity score of word in document, frequency of word in corpus, informativity score in corpus, (stress-relevant) word category.
* **Output**: 7 possible annotations {0,1,...,6}, representing relative stress prominence.

Training data not uploaded; consult Anntila et al. (2017) for further information on the annotation procedure.

Based on the following research:

Anttila, Arto, Timothy Dozat, Daniel Galbraith and Naomi Shapiro. 2017. Sentence stress in presidential speeches. *The 39th Annual Meeting of the DGfS Workshop on Prosody in Syntactic Encoding*, Saarbrücken, March 9, 2017.
