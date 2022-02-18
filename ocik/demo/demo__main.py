import sys

sys.path.append("./")
from ocik.demo import mutliapp, demo__intro, demo__parameter, demo__home, demo__learner, demo__explain

app = mutliapp.MultiApp()
app.add_app("Home", demo__home.app)
app.add_app("Introduction", demo__intro.app)
app.add_app("Parameter Learning", demo__parameter.app)
app.add_app("Causal Learning", demo__learner.app)
app.add_app("Explain Event", demo__explain.app)
app.run()