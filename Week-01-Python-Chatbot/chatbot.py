knowledge_base = {
    "hello": "Hey! How can I help you?",
    "hi": "Hello! How can I help you today?",
    "hey": "Hey there! What's on your mind?",
    "how are you": "I'm just a bot, but I'm running great!",
    "what is your name": "I'm PyBot, your friendly chatbot.",
    "what can you do": "I can answer basic questions and have a simple conversation.",
    "bye": "Goodbye! Have a great day.",
    "help": "Try asking: hello, how are you, what is your name, what can you do.",
}

print("PyBot is running. Type 'exit' to quit.\n")

while True:
    raw_input_text = input("You: ")

    # Sanitize & normalize
    user_input = raw_input_text.lower().strip()

    # Kill command
    if user_input == "exit":
        print("Bot: Goodbye!")
        break

    # Intent matching
    response = knowledge_base.get(user_input, "I do not understand.")

    print(f"Bot: {response}\n")
