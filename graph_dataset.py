import json


class GraphDataset:
    def __init__(self):
        self.entities = {}
        self.relations = {}
        self.relation_types = {}
        self.graphs = {}

    def _get_id(self, k, D):
        if k not in D:
            D[k] = len(D)

        return D[k]

    def _get_entity_id(self, entity):
        return self._get_id(entity, self.entities)

    def _get_relation_id(self, relation):
        return self._get_id(relation, self.relations)

    def _get_relation_type_id(self, relation_type):
        return self._get_id(relation_type, self.relation_types)

    def _get_graph_id(self, graph):
        return self._get_id(graph, self.graphs)

    def dumps(self):
        meta = {}
        meta["graphs"] = {v: list(k) for k, v in self.graphs.items()}
        meta["entities"] = {v: k for k, v in self.entities.items()}
        meta["relations"] = {v: k for k, v in self.relations.items()}
        meta["relation_types"] = {v: k for k, v in self.relation_types.items()}
        return json.dumps(meta)

    def dump(self, filename):
        with open(filename, 'w') as f:
            f.write(self.dumps())

    @classmethod
    def loads(cls, data):
        data = json.loads(data)
        self = cls()
        self.entities = data["entities"]
        self.relations = data["relations"]
        self.relation_types = data["relation_types"]
        self.graphs = data["graphs"]
        return self

    @classmethod
    def load(cls, filename):
        with open(filename) as f:
            return cls.loads(f.read())

    def _get_link(self, idx):
        e1, e2, r = self.relations[str(idx)]
        return self.entities[str(e1)], self.entities[str(e2)], self.relation_types[str(r)]

    def compress(self, G):
        # Assuming G list a list of string triples (e1, e2, r).
        new_G = frozenset(
            self._get_relation_id(
                (self._get_entity_id(e1),
                 self._get_entity_id(e2),
                 self._get_relation_type_id(r)))
            for e1, e2, r in G)
        return self._get_graph_id(new_G)

    def decompress(self, idx):
        return [list(self._get_link(link)) for link in self.graphs[str(idx)]]
