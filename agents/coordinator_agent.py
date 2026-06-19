from agents.scan_agent import ask_scan
from agents.atpg_agent import ask_atpg
from agents.sta_agent import ask_sta

def run_multi_agent(question, history=None):

    responses = []

    responses.append(
        ("SCAN", ask_scan(question, history))
    )

    responses.append(
        ("STA", ask_sta(question, history))
    )

    responses.append(
        ("ATPG", ask_atpg(question, history))
    )

    return responses