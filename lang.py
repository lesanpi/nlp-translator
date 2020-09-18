class Lang:
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.word2index = {'SOS': 0, "EOS": 1, "PAD": 2}
        self.wordCount = {}
        self.index2word = {0: 'SOS', 1: "EOS", 2: "PAD"}
        self.n_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.wordCount[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.wordCount[word] += 1

    def indexesFromSentence(self, sentence):
        return [self.word2index[word] for word in sentence.split(' ')]

    def sentenceFromIndex(self, index):
        return [self.index2word[ix] for ix in index]
