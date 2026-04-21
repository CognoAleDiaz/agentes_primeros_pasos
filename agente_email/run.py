"""
Entry point for the Daily Special Days Email Agent.
Runs the agent every day at 12:00pm using the `schedule` library.

Usage:
    python agente_email/run.py          # Starts the scheduler (runs forever)
    python agente_email/run.py --now    # Run once immediately (for testing)
"""

import sys
from datetime import datetime

import schedule
import time

from agent import build_graph


def get_today_formatted() -> str:
    """
    Return today's date in a human-readable Spanish format.
    Example: '16 de abril de 2026'
    """
    # Spanish month names (avoids locale issues on different machines)
    months = {
        1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
        5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
        9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
    }
    today = datetime.now()
    return f"{today.day} de {months[today.month]} de {today.year}"


def run_agent():
    """Build the graph, run it with today's date, and print the email draft."""
    graph = build_graph()

    date_str = get_today_formatted()

    result = graph.invoke({
        "date": date_str,
        "search_results": "",
        "selected_days": "",
        "email_draft": "",
    })


if __name__ == "__main__":
    # --now flag: run once immediately and exit (useful for testing)
    if "--now" in sys.argv:
        run_agent()
    else:
        # Schedule the agent to run every day at 12:00pm
        schedule.every().day.at("10:00").do(run_agent)
        print("⏰  Agente programado para las 12:00pm cada día. Esperando...")
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
