# This script is used for generating data from standard ctb.

from __future__ import print_function
from __future__ import division

from collections import defaultdict
import chardet


class PhraseTree(object):
    puncs = [",", ".", ":", "``", "''", "PU"]  ## (COLLINS.prm)

    def __init__(
            self,
            symbol=None,
            children=[],
            sentence=[],
            leaf=None,
    ):
        self.symbol = symbol  # label at top node
        self.children = children  # list of PhraseTree objects
        self.sentence = sentence
        self.leaf = leaf  # word at bottom level else None

        self._str = None

    def __str__(self):
        if self._str is None:
            if len(self.children) != 0:
                childstr = ' '.join(str(c) for c in self.children)
                self._str = '({} {})'.format(self.symbol, childstr)
            else:
                self._str = '({} {})'.format(
                    self.sentence[self.leaf][1],
                    self.sentence[self.leaf][0],
                )
        return self._str

    def propagate_sentence(self, sentence):
        """
        Recursively assigns sentence (list of (word, POS) pairs)
            to all nodes of a tree.
        """
        self.sentence = sentence
        for child in self.children:
            child.propagate_sentence(sentence)

    def pretty(self, level=0, marker='  '):
        pad = marker * level

        if self.leaf is not None:
            leaf_string = '({} {})'.format(
                self.symbol,
                self.sentence[self.leaf][0],
            )
            return pad + leaf_string

        else:
            result = pad + '(' + self.symbol
            for child in self.children:
                result += '\n' + child.pretty(level + 1)
            result += ')'
            return result

    @staticmethod
    def parse(line):
        """
        Loads a tree from a tree in PTB parenthetical format.
        """
        line += " "
        sentence = []
        _, t = PhraseTree._parse(line, 0, sentence)

        if t.symbol == 'TOP' and len(t.children) == 1:
            t = t.children[0]

        return t

    @staticmethod
    def _parse(line, index, sentence):
        "((...) (...) w/t (...)). returns pos and tree, and carries sent out."

        assert line[index] == '(', "Invalid tree string {} at {}".format(line, index)
        index += 1
        symbol = None
        children = []
        leaf = None
        while line[index] != ')':
            if line[index] == '(':
                index, t = PhraseTree._parse(line, index, sentence)
                children.append(t)

            else:
                if symbol is None:
                    # symbol is here!
                    rpos = min(line.find(' ', index), line.find(')', index))
                    # see above N.B. (find could return -1)

                    symbol = line[index:rpos]  # (word, tag) string pair

                    index = rpos
                else:
                    rpos = line.find(')', index)
                    word = line[index:rpos]
                    sentence.append((word, symbol))
                    leaf = len(sentence) - 1
                    index = rpos

            if line[index] == " ":
                index += 1

        assert line[index] == ')', "Invalid tree string %s at %d" % (line, index)

        t = PhraseTree(
            symbol=symbol,
            children=children,
            sentence=sentence,
            leaf=leaf,
        )

        return (index + 1), t

    def left_span(self):
        try:
            return self._left_span
        except AttributeError:
            if self.leaf is not None:
                self._left_span = self.leaf
            else:
                self._left_span = self.children[0].left_span()
            return self._left_span

    def right_span(self):
        try:
            return self._right_span
        except AttributeError:
            if self.leaf is not None:
                self._right_span = self.leaf
            else:
                self._right_span = self.children[-1].right_span()
            return self._right_span

    def brackets(self, advp_prt=True, counts=None):

        if counts is None:
            counts = defaultdict(int)

        if self.leaf is not None:
            return {}

        nonterm = self.symbol
        if advp_prt and nonterm == 'PRT':
            nonterm = 'ADVP'

        left = self.left_span()
        right = self.right_span()

        # ignore punctuation
        while (
                        left < len(self.sentence) and
                        self.sentence[left][1] in PhraseTree.puncs
        ):
            left += 1
        while (
                        right > 0 and self.sentence[right][1] in PhraseTree.puncs
        ):
            right -= 1

        if left <= right and nonterm != 'TOP':
            counts[(nonterm, left, right)] += 1

        for child in self.children:
            child.brackets(advp_prt=advp_prt, counts=counts)

        return counts

    def phrase(self):
        if self.leaf is not None:
            return [(self.leaf, self.symbol)]
        else:
            result = []
            for child in self.children:
                result.extend(child.phrase())
            return result

    @staticmethod
    def load_treefile(fname):
        trees = []
        for line in open(fname):
            t = PhraseTree.parse(line)
            trees.append(t)
        return trees

    def extract_pos(self):
        if len(self.children) == 0:
            str = self.__str__()
            (tag,word) = str[1:-1].split(' ')
            cs = []
            for c in word.decode('utf-8'):
                cs.append(c)
            result = ''
            result += (cs[0] + ' ' + tag + ' NA\n')
            for c in cs[1:]:
                result += (c + ' NotStart ' + 'AP\n')
            return result
        else:
            str = ''
            for child in self.children:
                str += (child.extract_pos())
            return str


if __name__ == '__main__':
    filename = 'data/ctb.data/test.clean'
    trees = PhraseTree.load_treefile(filename)
    str = ''
    for t in trees:
        str += (t.extract_pos() + '\n')
    with open(filename + '.append','w+') as f:
        f.write(str.encode('utf-8'))
    #print (t.extract_pos())
