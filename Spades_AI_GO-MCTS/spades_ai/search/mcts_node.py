import math


class MCTSNode:
    __slots__ = ('visit_count', 'total_value', 'children')

    def __init__(self):
        self.visit_count = 0
        self.total_value = 0.0
        self.children: dict[int, MCTSNode] = {}

    @property
    def mean_value(self):
        return self.total_value / self.visit_count if self.visit_count else 0.0

    def update(self, value):
        self.visit_count += 1
        self.total_value += value

    def get_or_create_child(self, action):
        if action not in self.children:
            self.children[action] = MCTSNode()
        return self.children[action]

    def uct_score(self, parent_visits, C):
        if self.visit_count == 0:
            return float('inf')
        return self.mean_value + C * math.sqrt(math.log(parent_visits) / self.visit_count)

    def best_child_uct(self, C):
        return max(self.children, key=lambda a: self.children[a].uct_score(self.visit_count, C))

    def best_child_visits(self):
        return max(self.children, key=lambda a: self.children[a].visit_count)

    def apply_penalty(self, mu):
        self.total_value -= mu * self.visit_count
