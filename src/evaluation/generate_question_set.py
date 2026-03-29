import json
import re
import random

INPUT_PATH = "data/processed/vehicle_chunks.json"
OUTPUT_PATH = "data/question_set.json"
MAX_QUESTIONS = 60

def extract_fields(text):
    data = {}

    # horsepower
    hp = re.search(r"producing (\d+) horsepower", text)
    if hp:
        data["horsepower"] = hp.group(1)

    # MSRP
    price = re.search(r"\$(\d{1,3}(?:,\d{3})*)", text)
    if price:
        data["msrp"] = price.group(0)

    # city MPG
    city = re.search(r"(\d+) city MPG", text)
    if city:
        data["city_mpg"] = city.group(1)

    # highway MPG
    highway = re.search(r"(\d+) highway MPG", text)
    if highway:
        data["highway_mpg"] = highway.group(1)

    return data


def main():
    with open(INPUT_PATH, "r") as f:
        chunks = json.load(f)

    questions = []
    qid = 1
    reverse_hp_map = {}
    all_titles = []

    for chunk in chunks:
        title = chunk["title"]
        text = chunk["text"]

        fields = extract_fields(text)

        all_titles.append(title)

        if "horsepower" in fields:
            hp = fields["horsepower"]
            reverse_hp_map.setdefault(hp, []).append(title)
            questions.append({
                "id": qid,
                "question": f"What is the horsepower of {title}?",
                "ground_truth": f"{fields['horsepower']} horsepower",
                "type": "fact",
                "reference_doc_id": chunk["doc_id"]
            })
            qid += 1

        if "msrp" in fields and "horsepower" in fields:
            questions.append({
                "id": qid,
                "question": f"What is the MSRP of {title} with {fields['horsepower']} horsepower?",
                "ground_truth": fields["msrp"],
                "type": "fact",
                "reference_doc_id": chunk["doc_id"]
            })
            qid += 1
    
    # Reverse Lookup Questions
    for hp, titles in reverse_hp_map.items():
        if len(titles) == 1:
            questions.append({
                "id": qid,
                "question": f"Which vehicle produces {hp} horsepower?",
                "ground_truth": titles[0],
                "type": "reverse",
                "reference_doc_id": chunk["doc_id"]
            })
            qid += 1
    
    # Simple Filtering Questions
    for _ in range(20):
        chunk = random.choice(chunks)
        title = chunk["title"]
        text = chunk["text"]

        fields = extract_fields(text)

        if "horsepower" in fields and "msrp" in fields:
            questions.append({
                "id": qid,
                "question": f"Which vehicle has {fields['horsepower']} horsepower and costs {fields['msrp']}?",
                "ground_truth": title,
                "type": "filter",
                "reference_doc_id": chunk["doc_id"]
            })
            qid += 1

    # Hallucination Questions
    hallucination_questions = [
    "What is the battery capacity of the 2014 Kia Soul?",
    "Which vehicle in the dataset is electric?",
    "What is the top speed of the 2017 Nissan Versa?",
    "Which car has autonomous driving capability?",
    "What is the engine torque of the 2015 Kia Soul?"
    ]

    for q in hallucination_questions:
        questions.append({
            "id": qid,
            "question": q,
            "ground_truth": "NOT_FOUND",
            "type": "hallucination",
            "reference_doc_id": chunk["doc_id"]
        })
        qid += 1

    # Randomly sample questions if we have more than MAX_QUESTIONS
    fact_q = []
    reverse_q = []
    filter_q = []
    halluc_q = []

    for q in questions:
        if "Which vehicle produces" in q["question"]:
            reverse_q.append(q)
        elif "Which vehicle has" in q["question"]:
            filter_q.append(q)
        elif q["ground_truth"] == "NOT_FOUND":
            halluc_q.append(q)
        else:
            fact_q.append(q)

    # balance
    final_questions = (
        fact_q[:15] +
        reverse_q[:15] +
        filter_q[:15] +
        halluc_q[:10]
    )

    random.shuffle(final_questions)
    questions = final_questions[:60]

    # fixing duplicates
    unique_questions = {}

    for q in questions:
        key = (q["question"], q["ground_truth"])
        if key not in unique_questions:
            unique_questions[key] = q

    questions = list(unique_questions.values())

    with open(OUTPUT_PATH, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"Generated {len(questions)} questions.")

if __name__ == "__main__":
    main()