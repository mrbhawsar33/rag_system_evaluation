""" Chunk dataset creation for vehicle knowledge base--
    Strategy: treat each car = one paragraph separated by blank lines.
    So chunking = split by empty lines.
"""

import json
import re

input_file = "../../data/raw/vehicle_knowledge_base.txt"
output_file = "../../data/processed/vehicle_chunks.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = f.read()

# split by blank lines (each car)
paragraphs = [p.strip() for p in data.split("\n\n") if p.strip()]

chunks = []

for i, para in enumerate(paragraphs):

    # first sentence becomes title
    first_line = para.split("\n")[0]

    # extract title like: The 2011 BMW 1 Series M
    title = re.match(r"(The .*?) is", first_line).group(1)

    # create doc id
    doc_id = title.lower().replace(" ", "_") + f"_{i:05d}"

    chunk = {
        "doc_id": doc_id,
        "title": title,
        "text": para
    }

    chunks.append(chunk)

# save json
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print("Total chunks:", len(chunks))
print("Saved to:", output_file)