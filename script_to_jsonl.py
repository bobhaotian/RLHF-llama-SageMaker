import json

# Input and output file paths
input_file = 'summarized_answers.json'
output_file = 'train.jsonl'

# List to hold the formatted JSON objects
jsonl_lines = []

print(f"Processing {input_file}...")

try:
    # Read the source JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process each item in the JSON data
    for item in data:
        # Extract fields from the JSON structure
        instruction = item.get('Instruction', '')
        input_text = item.get('Input', '')
        output_text = item.get('Output', '')
        
        # Skip entries with empty output
        if output_text == '':
            continue
        
        # Create the JSONL object with the required format
        formatted_obj = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }
        jsonl_lines.append(json.dumps(formatted_obj))

except FileNotFoundError:
    print(f"Error: {input_file} not found")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error parsing {input_file}: {e}")
    exit(1)

# Write the formatted objects to a JSONL file
with open(output_file, 'w') as f:
    for line in jsonl_lines:
        f.write(line + '\n')

# Indicate the process completion and provide statistics
print(f"Processing completed. Converted {len(jsonl_lines)} entries.")
print(f"The JSONL file is available at: {output_file}")
