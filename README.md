# StressPredict-NN
Multi-class softmax classification for predicting sentence stress annotation, trained on data from US Presidential Inaugural Addresses with gold-standard human annotations. Uses Keras Sequential model.

Input features include: word index within sentence chunk, lexical stress {yes, no, ambiguous}, number of segments in word, number of syllables in word, number of stresses, part of speech, dependency, mean, frequency of word in document, informativity score of word in document, frequency of word in corpus, informativity score in corpus, (stress-relevant) word category.
