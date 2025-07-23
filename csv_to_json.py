import csv
import json

# Input and output file paths
input_file = 'questions_answers.csv'
output_file = 'questions_answers.json'

# List to hold the JSON objects
json_data = []

print(f"Processing {input_file}...")

try:
    # Read the CSV file
    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Process each row in the CSV
        for row in reader:
            # Create JSON object with the specified field mappings
            json_obj = {
                "prompt": row.get('Question', ''),
                "chosen": row.get('concise', ''),
                "rejected": row.get('step-by-step', '')
            }
            
            # Skip entries with empty essential fields
            if json_obj["prompt"] == '' or (json_obj["chosen"] == '' and json_obj["rejected"] == ''):
                continue
                
            json_data.append(json_obj)

except FileNotFoundError:
    print(f"Error: {input_file} not found")
    exit(1)
except Exception as e:
    print(f"Error processing {input_file}: {e}")
    exit(1)

# Write the JSON data to output file
try:
    with open(output_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
        
    print(f"Conversion completed successfully!")
    print(f"Converted {len(json_data)} entries.")
    print(f"JSON file saved as: {output_file}")
    
except Exception as e:
    print(f"Error writing to {output_file}: {e}")
    exit(1) 