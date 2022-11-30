from datetime import datetime


def append_to_report(text, directory="output/reproducibility/",
                     file='single_agent_' + str(datetime.now()) + '.txt'):  # TODO create constant for this one
    with open(f"{directory}{file}", 'a') as f:
        f.write(text)
