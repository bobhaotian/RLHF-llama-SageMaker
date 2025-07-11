import csv
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

input_csv = "questions_topics.csv"
output_csv = "questions_topics_output.csv"

prompt_template = '''
Answer the question with numbered steps that build the idea gradually.
Requirement: It must have at least three short steps.
It must have clear transitions.
It does not contain heavy maths.
Question: {question}
'''

def get_step_by_step_answer(question):
    prompt = prompt_template.format(question=question)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

with open(input_csv, "r", encoding="utf-8") as infile, open(output_csv, "w", newline='', encoding="utf-8") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    header = next(reader)
    if len(header) == 2:
        header.append("step-by-step")
    writer.writerow(header)
    for row in reader:
        field, question = row[:2]
        print(f"Answering: {question} (Field: {field})")
        answer = get_step_by_step_answer(question)
        writer.writerow([field, question, answer])
        # time.sleep(1)  # To avoid rate limits

# Replace the original file with the new one
# os.replace(output_csv, input_csv)

# print("All questions answered and saved to 'questions_topics.csv'.") 