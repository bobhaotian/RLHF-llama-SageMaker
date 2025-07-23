# make_rm_dataset.py
import json, pathlib, random
rows = json.load(open("questions_answers.json"))    # your original file
with open("rm.jsonl", "w") as out:
    for r in rows:
        ch, rj = (r["chosen"], r["rejected"])
        out.write(json.dumps({"prompt": r["prompt"],
                              "chosen": ch,
                              "rejected": rj}) + "\n") 