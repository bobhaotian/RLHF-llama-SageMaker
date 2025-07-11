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
Answer the question one tight paragraph.
Requiements: No steps
No bullet points
Less than 120 words
Still factually correct
Question: {question}
'''

def get_concise_answer(question):
    prompt = prompt_template.format(question=question)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

with open(input_csv, "r", encoding="utf-8") as infile, open(output_csv, "w", newline='', encoding="utf-8") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    header = next(reader)
    if len(header) == 3:
        header.append("concise")
    writer.writerow(header)
    for row in reader:
        field, question, step_answer = row[:3]
        print(f"Answering: {question} (Field: {field})")
        concise_answer = get_concise_answer(question)
        writer.writerow([field, question, step_answer, concise_answer])
        # time.sleep(1)  # To avoid rate limits

# Replace the original file with the new one
# os.replace(output_csv, input_csv)

# print("All questions answered and saved to 'questions_topics.csv'.") 