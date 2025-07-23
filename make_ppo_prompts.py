# make_ppo_prompts.py
import json

# Read the rm.jsonl file and extract prompts
with open("rm.jsonl", "r") as input_file, open("ppo_prompts.jsonl", "w") as output_file:
    for line in input_file:
        data = json.loads(line.strip())
        prompt_obj = {"prompt": data["prompt"]}
        output_file.write(json.dumps(prompt_obj) + "\n")

print("Successfully created ppo_prompts.jsonl with prompts only.") 