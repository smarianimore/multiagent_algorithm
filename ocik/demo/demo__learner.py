import streamlit as st
from graphviz import Digraph
import sys
from PIL import Image
import pandas as pd

sys.path.append("./")

from ocik import Room
from ocik import CausalLeaner


def app():
    st.write(r'''
    # Learn Causal Model
    ''')
    image = Image.open('ocik/demo/media/algo.png')
    st.image(image, use_column_width=True)
    st.write(r"""
    
    ## Intro
    Before going further, we will explain the intuition that led us to our solution. Our first intuition 
    was to test if a change on one variable has an influence (change in distribution) on other variables. If so, 
    we do not know whether this influence is direct or if it has influenced by an intermediate variable. To remove 
    this ambiguity we are block incrementally all the other node connected to the node on which we act. 
    
    ### Approach 1
    
    Our approach starts with a complete oriented graph, i.e. a graph where each node is connected to all 
    other nodes. Then we will try to eliminate one by one the arrows using our causality test. The first step will be 
    a direct influence test. Then it will be followed by conditional influence tests of several orders. An influence 
    test of order $k$ is made by conditioning on $k$ variables. Because the algorithm does not necessarily return an 
    acyclic graph, we will do cycle detection and drop some connections by removing arrows with the least significant 
    causal influence in each cycle. """)

    st.write(r"""
    #### Simple example
    """)
    image = Image.open('ocik/demo/media/learner1.png')
    st.image(image, use_column_width=True)
    st.write(r"""
    Run:
    """)
    room = Room()
    f = Digraph()
    f.edges(list(room.bn()))
    st.graphviz_chart(f.source)
    if st.button('Learn'):
        bn = room.get_network()
        # obs_data = bn.sample(5000)
        obs_data = pd.read_csv('tmp/room.csv')

        estimator = CausalLeaner(bn.nodes(), non_dobale=[], env=bn, obs_data=obs_data)
        model, track = estimator.learn(max_cond_vars=4, do_size=100, trace=True)
        for i, edges in enumerate(track):
            var = "order " + str(i) + " : " if i != len(track) - 1 else "final result:"
            st.write(var)
            f = Digraph()
            f.edges(edges)
            graph = st.graphviz_chart(f.source)

    st.write(r'''
    ### Approach 2: Non doable case
    
    In a simulation it may be possible to make do-operator for all variable but in real life that might be not 
    always possible. We won't be able to make a do-operation on whether it's the day or the night even if we 
    wanted to. 
    
    To decide what to do with non-decidable arrow, we extend the previous algorithm by adding a conditional 
    independence test that use observational data. This is test used in constraint methods for learning Bayesian 
    Network structure (PC algorithm). In PC algorithm, this test is used to build an undirected graph 
    that will be oriented later. Thus, extension have no sense of direction. We will use it on non decidable 
    arrow to decide if we keep them. \fig{fig:algo2} show the result of this algorithm. Orange arrows represent 
    arrows deleted by the conditional independence test. The arrows $L\rightarrow P$ and $T\rightarrow H$ were 
    not delete because the test find that the concerne variable are not independent but cannot say anything about 
    the direction of influence. ''')

    image = Image.open('ocik/demo/media/learner2.png')
    st.image(image, use_column_width=True)
    st.write(r"""
        Run:
        """)

    f = Digraph()
    f.edges(list(room.bn()))
    st.graphviz_chart(f.source)
    if st.button('Learn 2'):
        bn = room.get_network()
        # obs_data = bn.sample(5000)
        obs_data = pd.read_csv('tmp/room.csv')

        estimator = CausalLeaner(bn.nodes(), non_dobale=['L', 'T'], env=bn, obs_data=obs_data)
        model, track = estimator.learn(max_cond_vars=4, do_size=100, trace=True)
        for i, edges in enumerate(track):
            var = "order " + str(i) + " : " if i != len(track) - 1 else "final result after postprocessing:"
            st.write(var)
            f = Digraph()
            f.edges(edges)
            graph = st.graphviz_chart(f.source)


if __name__ == '__main__':
    app()
