from scipy.special import softmax
import pandas as pd

from ocik import BayesianNetwork
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation


def convert2pgm(bn: BayesianNetwork):
    network = BayesianModel(ebunch=bn.G.edges())

    cpds = []
    for node in bn.G.nodes():
        par, cpd_tmp = bn.get_cpd(node)
        cpd = cpd_tmp[[f'{node}=0', f'{node}=1']].values.T.tolist()
        if len(par) == 0:
            tab_cpd = TabularCPD(variable=node, variable_card=2,
                                 values=cpd)
        else:
            tab_cpd = TabularCPD(variable=node, variable_card=2,
                                 values=cpd,
                                 evidence=par,
                                 evidence_card=[2] * len(par))
        cpds.append(tab_cpd)

    network.add_cpds(*cpds)
    return network


def most_probable_explanation(evidence: dict, bn: BayesianNetwork):
    """
    Most Probable Explanation
    evidence: {node: evidence_value, ...}
    bn : our own bayesian network class
    """
    network = convert2pgm(bn)
    prob = network.predict_probability(pd.DataFrame({name: [val] for name, val in evidence.items()}))

    # 1 - all possible explanation
    remain_node = list(set(list(network.nodes())) - set(evidence.keys()))
    n_par = len(remain_node)
    n = 2 ** n_par  # number of combinations
    binaries = [f"%0{n_par}d" % int(format(i, f'#0{n}b')[2:]) for i in range(n)]  # write in binary
    combination = {remain_node[i]: [int(b[i]) for b in binaries] for i in range(len(remain_node))}
    marginal = pd.DataFrame(combination)

    # 2 - compute join probability of each explanantion
    col_name = [[f'{col}_{marginal[col][i]}' for col in marginal.columns] for i in marginal.index]
    mar_prob = [prob[col].values.sum() for col in col_name]
    marginal.insert(0, 'probability', mar_prob)

    # marginal.probability /= marginal.probability.sum()
    marginal.probability = softmax(marginal.probability)

    marginal.insert(0, 'state', [str(col) for col in col_name])

    return marginal[['state', 'probability']].sort_values(by=['probability'], ascending=False)


def belief_propagation(evidence: dict, bn: BayesianNetwork):
    """
    evidence: {node: evidence_value, ...}
    bn : our own bayesian network class
    """
    network = convert2pgm(bn)
    net_infer = BeliefPropagation(network)
    to_infer = list(set(network.nodes()) - set(evidence.keys()))

    prior = {}
    prior_dist = net_infer.query(variables=to_infer, evidence={}, joint=False)
    for factor in prior_dist.values():
        node = factor.variables[0]
        values = factor.values.tolist()
        prior[factor.variables[0]] = pd.DataFrame({node: [0, 1], "prob": values})

    post = {}
    post_dist = net_infer.query(variables=to_infer, evidence=evidence, joint=False)
    for factor in post_dist.values():
        node = factor.variables[0]
        values = factor.values.tolist()
        post[factor.variables[0]] = pd.DataFrame({node: [0, 1], "prob": values})

    return prior, post
