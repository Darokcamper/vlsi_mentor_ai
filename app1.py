from dotenv import load_dotenv
load_dotenv()

from agents.router_agent import route_question

from agents.scan_agent import ask_scan
from agents.atpg_agent import ask_atpg
from agents.edt_agent import ask_edt
from agents.jtag_agent import ask_jtag
from agents.mbist_agent import ask_mbist
from agents.ijtag_agent import ask_ijtag
from agents.wrapper_agent import ask_wrapper
from agents.occ_agent import ask_occ
from agents.sta_agent import ask_sta

from agents.general_agent import ask_general

agent_map = {
    "SCAN": ask_scan,
    "ATPG": ask_atpg,
    "EDT": ask_edt,
    "MBIST": ask_mbist,
    "JTAG": ask_jtag,
    "IJTAG": ask_ijtag,
    "WRAPPER": ask_wrapper,
    "OCC": ask_occ,
    "STA": ask_sta,
}

print("\n=== VLSI Mentor AI ===\n")

while True:

    question = input("\nAsk Question (or quit): ")

    if question.lower() == "quit":
        break

    topic = route_question(question)
    
    known_topics = [
        "SCAN",
        "ATPG",
        "EDT",
        "MBIST",
        "JTAG",
        "IJTAG",
        "WRAPPER",
        "OCC",
        "STA",
        "TCL",
        "LINUX"
    ]

    if topic not in known_topics:
        topic = "GENERAL"
    
    print(f"\n[Router] -> {topic}")

    agent = agent_map.get(topic, ask_general)

    try:
        answer = agent(question)

    except Exception as e:
        print(f"\nAgent Error: {e}")
        print("\nFalling back to General Agent...\n")

        answer = ask_general(question)

    print("\n")
    print(answer)
    print("\n" + "=" * 80)