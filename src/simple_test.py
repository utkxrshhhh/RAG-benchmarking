import json

# Open our questions file
with open("experiments/questions.json", "r") as file:
    questions = json.load(file)

# Print the questions
for question in questions:
    print(question["question"])