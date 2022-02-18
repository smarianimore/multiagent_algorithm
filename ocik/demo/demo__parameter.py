import streamlit as st
from graphviz import Digraph
import sys

from PIL import Image

sys.path.append("./")

from ocik.example import Asia


def app():
    st.write(r"""
    ## Learn parameter
    
    Let's imagine that we have built a Bayesian network representing our causal model, the next step consists in fill in 
    the conditional probability tables. One way to do this is to look for the best set of parameters to report the 
    observed data. 
    
    There are two approaches to calculating the parameters:
    
    - **Maximum likelihood**: which considers the parameters as constants and tries to determine them by maximizing the 
    likelihood. 
    
    - **The Maximum to Priori**: Uses the results of maximum likelihood to construct a posteriori distribution.
    
    
    Parameter learning can be resume to counting variable value given their parent value and compute a probability 
    from it. We improve this counting process with our do-operator. Let's take a simple example to illustrate how we 
    improve this learning through our data augmentation. """)

    image = Image.open('ocik/demo/media/parameter.png')
    st.image(image, use_column_width=True)


if __name__ == '__main__':
    app()
