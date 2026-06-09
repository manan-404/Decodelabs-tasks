# Week 01 — Python Chatbot

A rule-based command-line chatbot built in Python, implementing a full
Input-Process-Output (IPO) pipeline.

## How to Run

    python chatbot.py

No external libraries required. Python 3.x only.

## Concepts Demonstrated

- Infinite input loop using `while True` with a kill command (`exit`) to break cleanly
- Input sanitization and normalization via `.lower()` and `.strip()`
- Intent matching against a structured knowledge base (dictionary)
- Fallback response for unrecognized inputs

## Project Structure

    Week-01-Python-Chatbot/
    ├── chatbot.py
    └── README.md

## Sample Interaction

    You: Hello
    Bot: Hey! How can I help you?

    You: WHAT IS YOUR NAME
    Bot: I'm PyBot, your friendly chatbot.

    You: exit
    Bot: Goodbye!
