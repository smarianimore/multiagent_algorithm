import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator


class ConditionalProbability:
    def __init__(self, dataset, edges):
        self.dataset = dataset
        self.edges = edges
        self.model = BayesianNetwork(edges)
        self.estimator = BayesianEstimator(self.model, dataset)

    def get_node_prob(self, node):
        # Get the values of the probability for a specific node
        if node is not None:
            cpd_node = self.estimator.estimate_cpd(node, prior_type="K2").get_values()
            return cpd_node

    def get_network_prob(self):
        # Get the CPD table of the entire network
        cpd_tables = self.estimator.get_parameters(prior_type='K2')
        return cpd_tables


if __name__ == "__main__":
    dataset = pd.read_csv("../ocik/demo/store/test/network.csv", sep=',')
    edges = [("Pr", "L"), ("L", "Pow"), ("H", "Pow"), ("C", "Pow"), ("H", "T"), ("C", "T"),
            ("B", "W"), ("O", "T"), ("W", "T")]
    p = ConditionalProbability(dataset, edges).get_node_prob('T')
    print(p)



