from summary.bm25 import BM25
import collections
import jieba
import jieba
import re


class TextRankSentences(object):

    def __init__(self, doc):

        # 阻尼系数
        self.d = 0.85

        self.max_iter = 200
        self.min_diff = 0.001

        self.separator = "[，,。:：“”？?！!；;]"

        self.D = len(doc)

        self.top = {}
        self.weight = {}
        self.weight_sum = {}
        self.vertex = {}

        self.bm = BM25(doc)
        self.doc = doc

        cnt = 0

        for sentence in doc:
            scores = self.bm.sim_all(sentence)
            self.weight[cnt] = scores
            self.weight_sum[cnt] = sum(scores) - scores[cnt]
            self.vertex[cnt] = 1.0
            cnt += 1

        for every_iter in range(self.max_iter):
            m = {}
            max_diff = 0
            for i in range(self.D):
                m[i] = 1 - self.d
                for j in range(self.D):
                    if j == i or self.weight_sum[j] == 0:
                        continue
                    m[i] += (self.d * self.weight[j][i] / self.weight_sum[j] * self.vertex[j])

                diff = abs(m[i] - self.vertex[i])

                if diff > max_diff:
                    max_diff = diff

            self.vertex = m

            if max_diff <= self.min_diff:
                break

        for i in range(self.D):
            # self.top[self.vertex[i]] = i
            self.top[i] = self.vertex[i]

    def get_top_n(self, size=5):

        res = sorted(self.top, key=lambda x: self.top[x])
        res.reverse()

        result = []
        result.append(res[0])
        for i in res:
            # print("-----------------------")
            value = 0
            for j in result:
                # print(i, j, self.doc[i], self.doc[j])
                value = self.bm.sim(self.doc[i], j)
                # print(value)
                if value > 10:
                    break
            # print("-----------------------")
            if value <= 10 and i not in result:
                result.append(i)

            if len(result) >= size:
                break

        result4see = "".join(["".join(self.doc[i]) for i in sorted(result)])
        return result4see


class TextParser(object):

    def __init__(self):
        self.stop_words = []
        with open('../dic/stop_words.txt', 'r', encoding='utf-8') as f:
            for line in f:
                self.stop_words.append(line.strip())
        self.stop_words.append(' ')
        self.stop_words.append('\n')
        self.stop_words.append('\t')

        self.delimiter = frozenset(u'。！？； >【】')
        # delimiters = frozenset(u'。！？；>【】')

    def set_delimiter(self, delimiter):
        self.delimiter = frozenset(delimiter)

    def cut_sentence(self, sentence):

        buf = []
        for ch in sentence:
            buf.append(ch)
            if self.delimiter.__contains__(ch):
                yield ''.join(buf)
                buf = []
        if buf:
            yield ''.join(buf)

    def generate_docs(self, doc):
        docs = self.cut_sentence(doc)
        docs = [jieba.lcut(i) for i in docs]
        return docs
