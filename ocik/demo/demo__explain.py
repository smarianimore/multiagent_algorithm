import streamlit as st
from graphviz import Digraph
import sys
from PIL import Image
import altair as alt

sys.path.append("./")

from ocik.example import Room, RoomBase
from ocik import most_probable_explanation, belief_propagation


def app():
    st.write(r'''
    ## Explain Events
    ### Most Probable Explanation
    
    Most probable explanation (MPE), also known as max propagation, computes the most probable configuration of 
    variables that do not have evidence : 
    
    $MPE = \underset{X \in \mathcal{X}\setminus \mathbf{e}}{argmax} \mathbb{P}(X \mid \mathbf{e})$
    
    This is a combinatorial optimization problem. The simplest way to solve it is to do exhaustive search in all 
    possible state and compute each state probability in order to find the most probable one. We did it for this 
    small network : 
    ''')

    # room = Room()
    room = RoomBase()
    f = Digraph()
    f.edges(list(room.bn()))
    st.graphviz_chart(f.source)

    st.write(r"""
    choose on node and add evidence on it
    """)
    node = st.selectbox("mpe, node", room.bn.nodes())
    val = st.radio("mpe, value", (0, 1))
    if st.button('mpe, explain !'):
        explanation = most_probable_explanation({node: int(val)}, room.bn)

        bars = alt.Chart(explanation).mark_bar().encode(
            x='probability',
            y="state"
        )

        text = bars.mark_text(
            align='left',
            baseline='middle',
            dx=3  # Nudges text to right so it doesn't appear on top of the bar
        ).encode(
            text='probability:O'
        )

        plot = (bars + text).properties(height=500, width=700)
        st.write(plot)

    st.write(r"""
    ### Belief propagation
    
    BBN can be used to identify the possible root cause(s) of an event. Setting evidence on a variable may affect the 
    probability distribution of the others variables after propagating backwards against the direction of query 
    variable. In this way, BBN is able to diagnose the possible root cause(s) and influence by identifying the change 
    of posterior probability of the ancestor nodes. We this on our previous example.""")

    image = Image.open('ocik/demo/media/bp.png')
    st.image(image, use_column_width=True)

    st.write(r"""
        choose on node and add evidence on it
        """)

    node = st.selectbox("bp, node", room.bn.nodes())
    val = st.radio("bp, value", (0, 1))
    if st.button('bp, explain !'):
        prior, post = belief_propagation({node: int(val)}, room.bn)
        st.write("prior distribution")
        colprior = st.beta_columns(len(prior))
        for i, (k, df) in enumerate(prior.items()):
            bars = alt.Chart(prior[k]).mark_bar().encode(
                x=alt.X(k, scale=alt.Scale(domain=[0, 1.5])),
                y=alt.Y('prob', scale=alt.Scale(domain=[0, 1])),
            )
            text = bars.mark_text(
                align='center',
                baseline='middle',
                dy=-10  # Nudges text to right so it doesn't appear on top of the bar
            ).encode(
                text=alt.Text('prob', format=",.2f")
            )

            plot = (bars+text).properties(height=200, width=200)

            colprior[i].write(plot)

        st.write("distribution after belief propagation")
        colpost = st.beta_columns(len(post))
        for i, (k, df) in enumerate(post.items()):
            bars = alt.Chart(post[k]).mark_bar().encode(
                x=alt.X(k, scale=alt.Scale(domain=[0, 1.5])),
                y=alt.Y('prob', scale=alt.Scale(domain=[0, 1])),
                color=alt.value("#27ae60"))

            text = bars.mark_text(
                align='center',
                baseline='middle',
                dy=-10  # Nudges text to right so it doesn't appear on top of the bar
            ).encode(
                text=alt.Text('prob', format=",.2f")
            )

            plot = (bars+text).properties(height=200, width=200)

            colpost[i].write(plot)

if __name__ == '__main__':
    app()
