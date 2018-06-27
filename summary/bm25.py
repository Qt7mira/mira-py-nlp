import math


class BM25(object):

    def __init__(self, doc, mode):
        self.D = 1
        self.avg_dl = 0

        self.k1 = 1.5
        self.b = 0.75

        self.f = {}
        self.df = {}
        self.idf = {}
        self.doc = []

        self.doc = doc
        self.D = len(self.doc)
        for sentence in self.doc:
            self.avg_dl += len(sentence)

        self.avg_dl /= self.D

        # mode值有两个：1，2。模式1为原始bm25算法。模式2为bm25进行了简单的公式调整。
        self.mode = mode

        index = 0

        for sentence in self.doc:
            tf = {}
            for word in sentence:
                if word in tf.keys():
                    tf[word] += 1
                else:
                    tf[word] = 1
            self.f[index] = tf

            for k, v in tf.items():
                if k in self.df.keys():
                    self.df[k] += 1
                else:
                    self.df[k] = 1

            index += 1

        for k, v in self.df.items():
            self.idf[k] = math.log(self.D - v + 0.5) - math.log(v + 0.5)

    def sim(self, sentence, index):
        score = 0.0
        # print("---------------")
        # print(sentence)
        # print(index)
        # print(self.f[index])
        # print("---------------")
        for word in sentence:
            if index not in self.f.keys():
                continue
            d = len(self.doc[index])
            if word in self.f[index].keys():
                wf = self.f[index][word]
            else:
                wf = 0
            score += self.idf[word] * wf * (self.k1 + 1) / (wf + self.k1 * (1 - self.b + self.b * d / self.avg_dl))

        if self.mode == 1:
            return score
        elif self.mode == 2:
            return score / (len(sentence) + len(self.doc))

    def sim_all(self, sentence):
        scores = {}
        for i in range(self.D):
            scores[i] = self.sim(sentence, i)
        return scores
