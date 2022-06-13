# SM NOTES

Built on Kanvaly's original repo forked at [https://github.com/smarianimore/causality_detection](https://github.com/smarianimore/causality_detection)

Within `utils`:
 - `config.py` should be changed according to the scenario actually simulated through iCasa, and also stores learning params, and intervention TCP socket address
 - `drawing.py` must be configured with the correct path to graphviz binaries
 - `probabilityEstimation` computes probabilities based on observational data and parents relationships in the causal net

Main components:
 - `agent.py`: represents the learning agent, hence encapsulates [Kanvaly's learning algorithm](https://github.com/smarianimore/causality_detection) and implements the request-response protocol for multi-agent learning
   - param `edges` decides whether learning is from scratch or incrementally built on previous knowledge
 - `example.py`: used only for random networks (scenarios), to launch single-agent experiments, relying on the networks (scenarios) defined in `ocik/example.py`
 - `ocik/causal_learner.py`: Kanavly's algo modified to choose offline/online mode
 - `online.py`: handles interventions request to [iCasa side](https://github.com/smarianimore/iCasa)
 - `test.py`: single agent experiments
 - `run.py`: two agents experiments (**second one** is the one that asks for help, atm)
