import streamlit as st
import sys
from PIL import Image

sys.path.append("./")


def app():
    st.write(r'''
    ## Home Page
    ''')

    image = Image.open('ocik/demo/media/home.png')
    st.image(image, use_column_width=True)

    st.write(r"""
    # Abstract
    
    Anomaly detection is a topic that generates a lot of interest in the ML community.  Once an anomaly 
    has been detected, the new concerne is to trace its source. This study aims to determine the cause of anomalies 
    in a smart home scenario. First, we first learn the causal representation of the smart home using an innovative 
    approach that combines interaction with the environment and observation data. Then we embed the resulting causal 
    model into a Bayesian network. Finaly we use bayesian network to do belief propagation in order to find possible 
    cause of abnormal event given some evidence. """)


if __name__ == '__main__':
    app()
