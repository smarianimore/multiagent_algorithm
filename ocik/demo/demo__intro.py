import sys

sys.path.append("./")

import streamlit as st


def app():
    st.write(r"""

# INTRODUCTION

The challenge of this project is to predict the cause of anomalies that could occur in a smart home. Starting from 
the anomalies, we want to build a system that finds what caused them. One way to approach this problem is to consult 
an expert to hard-code the cause and effect relationships between the objects in our smart house. However, 
this approach lacks flexibility because we should know in advance the impact each variable would have with the 
others. 

Let's take a light bulb that heat up a lot and a temperature sensor as an example. One would think at 
first sight that this bulb does not have a great effect on the temperature of the house and therefore has no causal 
relation with the temperature sensor. Now imagine that this sensor is placed next to the light bulb. This will 
completely falsify the sensor measurement. A hard-coded system would not be able to see such a relationship in its 
diagnosis and will certainly blame the heater.
 

The first step to solve this problem is to build the causal model of the environment. Bayesian Network(BN) is a good 
candidate since he can learn relationships of the domain component using only data. He makes it possible to represent 
the dependencies/independencies in probabilities of several variables. But the problem here is that bayesian networks 
are not necessarily causal . However, any causal model can be implemented in the form of a Bayesian network. The 
interpretation that we make of an arrow in a causal diagram ($A \rightarrow B : $A influence B) is not the same as 
what we give to Bayesian network arrows (if we see A, we will probably see B but we don't know if it's A that make B 
appear). So the correctness of the Bayesian Network causal model needs to be carefully verified. 
 
 
There are two major tasks in learning a BN, learning the graphical structure and then 
learning the parameters (i.e., conditional probability table entries). The majority of the methods presented in the 
literature learn Bayesian networks using correlation from historical data . However, looking for causal relationships 
only with observational data can lead us to consider many false positives relations.


An example in the case of the smart house would be to look at the data and see that every time the television is on, 
you have the light bulb in the room that is also on. The model would think that it's the television that turn the 
light bulb on. But, the real reason is that there is a hidden variable, for example a person in the room who causes 
both to be turned on. Thus, we can see that we need more than observation to conclude that there is a causal 
relationship between variables. A simple solution to that is to have interaction with the environment. The 
intervention here will consist of acting on the environment, and see what this will lead to. We will turn off the 
television and see if this will also turn off the light. Then realise that this is not the case, which may mean that 
the hypothesis of causal relationship is wrong. 


This paper proposes an innovative way of learning  causal model that use interaction with the environment when it is 
possible and observational data. Then we embed our causal model into a Bayesian Network to do backwards reasoning in 
order to find cause of event such as anomalies. 

""")

if __name__ == '__main__':
    app()