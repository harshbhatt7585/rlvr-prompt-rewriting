from datasets import load_dataset, Dataset, DatasetDict

dataset_name = "ttbui/html_alpaca"

dataset = load_dataset(dataset_name)

new_dataset_list = []

for data in dataset['train']:
    instruction = data['instruction']
    output = data['output']

    prompt = f"""Rewrite the following prompt to add fine details for world class output.

Do the following steps:
1. Analyze the prompt and understand the user's intent.
2. Add fine details to the prompt to make it more specific and detailed.
3. If required add styling, colours, fonts, layouts, animations, etc. to the prompt to more aesthetic and visually appealing output.
4. Return the rewritten prompt in the json format.

Prompt:
{instruction}
"""

    # Optionally keep the original output for reference
    new_dataset_list.append({
        "original_instruction": instruction,
        "original_output": output,
        "rewriting_prompt": prompt
    })

# Convert list to HuggingFace Dataset object
new_hf_dataset = Dataset.from_list(new_dataset_list)
# Optionally wrap in DatasetDict for train split, for easier downstream use
new_hf_dict = DatasetDict({"train": new_hf_dataset})
# Save to disk at ./data
new_hf_dict.save_to_disk("./dataset")