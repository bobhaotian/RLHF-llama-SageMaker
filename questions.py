from openai import OpenAI
import time
import csv
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

topics = [
    "Linear Algebra",
    "Calculus",
    "Probability Theory",
    "Statistics",
    "Graph Theory",
    "Number Theory",
    "Combinatorics",
    "Real Analysis",
    "Complex Analysis",
    "Abstract Algebra",
    "Differential Equations",
    "Topology",
    "Numerical Methods",
    "Optimization",
    "Set Theory",
    "Discrete Mathematics",
    "Machine Learning",
    "Artificial Intelligence",
    "Algorithms",
    "Data Structures",
    "Cryptography",
    "Computational Geometry",
    "Operating Systems",
    "Computer Networks",
    "Database Systems",
    "Software Engineering",
    "Theory of Computation",
    "Formal Languages and Automata",
    "Computer Graphics",
    "Parallel Computing"
]

all_questions = set()

def print_runtime(start_time):
    end_time = time.time()
    elapsed = end_time - start_time
    mins, secs = divmod(elapsed, 60)
    print(f"Total runtime: {int(mins)} min {secs:.2f} sec")

# Record start time
start_time = time.time()

for topic in topics:
    print(f"Generating questions for topic: {topic}...")
    prompt = f"""
    Generating questions for topic: {topic}...
    Generate a list of 100 questions that each ask to explain a technical term specifically from the field of {topic}.
    Use only common jargon typical to {topic}. All questions should be concise and the concepts should be simple. 
    Just write the questions. Do not write in the form a number list. 
    For example, if the topic is Data Structure. The questions should be like:
    "What is a linked list?"
    "Describe a stack data structure."
    "Why do we use hash tables?"
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    lines = response.choices[0].message.content.strip().split("\n")
    for line in lines:
        cleaned_line = line.strip().strip('"')
        if cleaned_line:
            all_questions.add((topic, cleaned_line))
    time.sleep(1)

# Write to CSV
with open("questions_topics.csv", "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Topic", "Question"])
    for topic, question in sorted(all_questions):
        writer.writerow([topic, question])

print(f"Done. Total unique questions: {len(all_questions)}. Saved to 'questions_topics.csv'.")
print_runtime(start_time)
