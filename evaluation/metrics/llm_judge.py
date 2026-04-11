import argparse
import json
import re
from collections import defaultdict

import numpy as np
from openai import OpenAI

from mem0.memory.utils import extract_json
from src.runtime_config import apply_runtime_env, load_runtime_config

try:
    apply_runtime_env(load_runtime_config())
except Exception:
    pass

client = OpenAI(api_key=None if not __import__("os").getenv("OPENAI_API_KEY") else __import__("os").getenv("OPENAI_API_KEY"))

ACCURACY_PROMPT = """
Your task is to label an answer to a question as CORRECT or WRONG. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a gold (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return only one label: CORRECT or WRONG.
Do not add explanations, punctuation, or any other text.
"""


def _sanitize_text(value):
    """Remove characters that can break downstream API serialization."""
    text = str(value)
    text = text.replace("\x00", "")
    return text.encode("utf-8", "replace").decode("utf-8")


def evaluate_llm_judge(question, gold_answer, generated_answer):
    """Evaluate the generated answer against the gold answer using an LLM judge."""
    prompt = ACCURACY_PROMPT.format(
        question=_sanitize_text(question),
        gold_answer=_sanitize_text(gold_answer),
        generated_answer=_sanitize_text(generated_answer),
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
    )
    content = (response.choices[0].message.content or "").strip()
    normalized = content.upper()
    if normalized == "CORRECT":
        label = "CORRECT"
    elif normalized == "WRONG":
        label = "WRONG"
    else:
        matches = re.findall(r"\b(CORRECT|WRONG)\b", normalized)
        if matches:
            label = matches[-1]
        else:
            label = json.loads(extract_json(content))["label"].upper()
    return 1 if label == "CORRECT" else 0


def main():
    """Main function to evaluate RAG results using LLM judge."""
    parser = argparse.ArgumentParser(description="Evaluate RAG results using LLM judge")
    parser.add_argument(
        "--input_file",
        type=str,
        default="results/default_run_v4_k30_new_graph.json",
        help="Path to the input dataset file",
    )

    args = parser.parse_args()

    dataset_path = args.input_file
    output_path = f"results/llm_judge_{dataset_path.split('/')[-1]}"

    with open(dataset_path, "r") as f:
        data = json.load(f)

    LLM_JUDGE = defaultdict(list)
    RESULTS = defaultdict(list)

    index = 0
    for k, v in data.items():
        for x in v:
            question = x["question"]
            gold_answer = x["answer"]
            generated_answer = x["response"]
            category = x["category"]

            # Skip category 5
            if int(category) == 5:
                continue

            # Evaluate the answer
            label = evaluate_llm_judge(question, gold_answer, generated_answer)
            LLM_JUDGE[category].append(label)

            # Store the results
            RESULTS[index].append(
                {
                    "question": question,
                    "gt_answer": gold_answer,
                    "response": generated_answer,
                    "category": category,
                    "llm_label": label,
                }
            )

            # Save intermediate results
            with open(output_path, "w") as f:
                json.dump(RESULTS, f, indent=4)

            # Print current accuracy for all categories
            print("All categories accuracy:")
            for cat, results in LLM_JUDGE.items():
                if results:  # Only print if there are results for this category
                    print(f"  Category {cat}: {np.mean(results):.4f} ({sum(results)}/{len(results)})")
            print("------------------------------------------")
        index += 1

    # Save final results
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=4)

    # Print final summary
    print("PATH: ", dataset_path)
    print("------------------------------------------")
    for k, v in LLM_JUDGE.items():
        print(k, np.mean(v))


if __name__ == "__main__":
    main()
