import numpy as np

class Cluster():
    def __init__(self, members, market):
        self.members = members
        self.market = market

    def add_to_cluster(self, new_member):
        self.members += [new_member]
        new_member.in_cluster = self

    def activate(self):
        """
        Sets all probabilities to 0 or 1
        """
        cluster_P = np.random.choice([0,1])
        for member in self.members:
            member.Pi = cluster_P

    def self_destruct(self):
        """
        Resets probabilities of members and self destructs
        """
        for member in self.members:
            member.Pi = 0.5
            member.in_cluster = None

        self.market.clusters.remove(self)
