from math import log as math_log
import re
from itertools import chain
from sys import version_info
import os

is_python2 = (version_info.major == 2)

if not is_python2:
    basestring = str

run_path = os.path.split(os.path.realpath(__file__))[0]


def log(x):
    if x == 0:
        return -100
    else:
        return math_log(x)


class Trie:
    """定义基本的Trie树结构，便于存储词典（词+词频）。
    主要的代码量是基于Python类的特殊方法来定义一些函数，
    达到表面上看起来和dict的使用方法基本一致的效果。
    """

    def __init__(self, path_or_trie=None):
        self.dic = {}
        self.end = True
        self.num_items = 0  # 总词数
        self.total_items = 0  # 总词频
        self.log_total_items = log(self.total_items)  # 对数总词频
        if isinstance(path_or_trie, basestring):  # 从文件中加载，文件的每一行是“词 词频
            with open(path_or_trie) as f:
                for l in f:
                    l = re.split(' +', l.strip())
                    if is_python2:
                        self.__setitem__(l[0].decode('utf-8'), int(l[1]))
                    else:
                        self.__setitem__(l[0], int(l[1]))
        elif path_or_trie is not None:
            self.update(path_or_trie)

    def __setitem__(self, item, count):
        if count == 0:  # 设置词频为零则相当于删除该词
            return self.__delitem__(item)

        _ = self.dic
        for c in item:
            if c not in _:
                _[c] = {}
            _ = _[c]

        if self.end in _:  # 调整词频
            self.total_items += (count - _[self.end][1])
        else:  # 增加新词
            self.total_items += count
            self.num_items += 1

        _[self.end] = (item, count)
        self.log_total_items = log(self.total_items)  # 更新对数词频

    def __getitem__(self, item):  # 获取指定词的频率，不存在则返回0
        _ = self.dic
        for c in item:
            if c not in _:
                return 0
            _ = _[c]

        return _.get(self.end, ('', 0))[1]

    def __delitem__(self, item):  # 删除某个词
        _ = self.dic
        for c in item:
            if c not in _:
                return None
            _ = _[c]

        if self.end in _:
            self.num_items -= 1
            self.total_items -= _[self.end][1]
            del _[self.end]

    def __iter__(self, _=None):  # 以(词, 词频)的形式逐一返回所有记录
        if _ is None:
            _ = self.dic

        for c in _:
            if c == self.end:
                yield _[self.end]
            else:
                for i in self.__iter__(_[c]):
                    yield i

    def __str__(self):  # 方便调试的显示
        return '<Trie: %s items, %s frequency>' % (self.num_items,
                                                   self.total_items)

    def __repr__(self):
        return self.__str__()

    def search(self, sent):  # 返回字符串中所有能找到的词语
        result = {}  # 结果是{(start, end): (词, 词频)}的字典
        for i, c1 in enumerate(sent):
            _ = self.dic
            for j, c2 in enumerate(sent[i:]):
                if c2 in _:
                    _ = _[c2]
                    if self.end in _:
                        result[i, i + j + 1] = _[self.end]
                else:
                    break

        return result

    def update(self, tire):  # 用一个词典更新当前trie树
        for i, j in tire:
            self.__setitem__(i, j)

    def get_proba(self, w, logit=True):  # 算词频
        _ = self.__getitem__(w)
        if logit:
            return log(_) - self.log_total_items
        else:
            return _ / self.total_items


class DAG:
    """定义一般的有向无环图（Directed Acyclic Graph）对象，
    便于在各种场景下使用。其中optimal_path方法使用viterbi
    算法来给出最优路径。
    """

    def __init__(self, nb_node, null_score=-100):
        self.edges = {}
        self.nb_node = nb_node
        self.null_score = null_score

    def __setitem__(self, start_end, score):  # 构建图上的加权边
        start, end = start_end  # key是(start, end)下标对
        if start not in self.edges:
            self.edges[start] = {}
        self.edges[start][end] = score

    def optimal_path(self):
        """动态规划求最优路径
        result的key是当前字的下标，代表截止到前一字的规划结果，
        result的第一个值是list，表示匹配片段的(start, end)下标对；
        result的第二个值是路径的分数
        """
        result = {0: ([], 1)}
        start = 0  # 当前字的下标
        length = self.nb_node
        while start < length:
            if start in self.edges:  # 如果匹配得上
                for i, j in self.edges[start].items():  # 这里i是终止下标
                    score = result[start][1] + j  # 当前路径分数
                    # 如果当前路径不在result中，或者它的分数超过已有路径，则更新
                    if i not in result or (score > result[i][1]):
                        result[i] = result[start][0] + [(start, i)], score

            # 为了下一步的匹配，如果下一字还不在result中，
            # 就按单字来插入，概率为null_score
            if start + 1 not in result:
                score = result[start][1] + self.null_score
                result[start
                       + 1] = result[start][0] + [(start, start + 1)], score

            start += 1

        return result[self.nb_node][0]

    def _all_paths(self, n):  # all_paths的辅助函数，递归获取从n开始的所有路径
        if n in self.edges:  # 如果成立则意味着n还不是终点
            paths = []
            for m in self.edges[n]:
                paths.extend([[n] + _ for _ in self._all_paths(m - 1)])
        else:  # 意味着n是终点
            paths = [[n]]

        return paths

    def all_paths(self):  # 返回所有连通路径（包括孤立节点）
        ends = set(chain(*self.edges.values()))
        starts = [n for n in range(self.nb_node) if n + 1 not in ends]
        paths = []

        for n in starts:
            paths.extend(self._all_paths(n))

        return paths
