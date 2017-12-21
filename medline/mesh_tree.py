from os.path import dirname, join
import networkx as nx

MESH_TREE_FILE = join(dirname(__file__), 'mtrees2017.bin')


def create_edgelist(numbers):
    edgelist = set()

    for number in numbers:
        indices = [1]

        for i, x in enumerate(number):
            if x == '.':
                indices.append(i)

        indices.append(len(number))

        for i in range(1, len(indices)):
            u = number[:indices[i]]
            v = number[:indices[i - 1]]

            edgelist.add((u, v))

    return list(edgelist)


class MeshTree(object):

    def __init__(self):
        terms, numbers = zip(*[l.rstrip('\n').split(';')
                               for l in open(MESH_TREE_FILE)])

        edgelist = create_edgelist(numbers)
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(edgelist)

        terms = [t.lower() for t in terms]
        self.number_term = dict(zip(numbers, terms))

        self.term_numbers = {}
        for term, number in zip(terms, numbers):
            if term not in self.term_numbers:
                self.term_numbers[term] = [number]
            else:
                self.term_numbers[term].append(number)

    def _get_descendant_numbers(self, term):
        term = term.lower()
        desc_numbers = set()

        try:
            term_numbers = self.term_numbers[term]
        except KeyError:
            return desc_numbers

        for number in term_numbers:
            desc_numbers.update(nx.descendants(self.graph, number))

        return desc_numbers

    def get_categories(self, term):
        numbers = self._get_descendant_numbers(term)

        categories = [number for number in numbers
                      if self.graph.out_degree(number) == 0]

        return categories
