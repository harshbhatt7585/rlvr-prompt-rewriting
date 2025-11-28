from datasets import load_dataset

dataset_name = "ttbui/html_alpaca"

dataset = load_dataset(dataset_name)


new_dataset = []

for data in dataset['train']:
    instruction = data['instruction']
    output = data['output']

    prompt = """Rewrite the following prompt to add fine details for world class output.

Do the following steps:
1. Analyze the prompt and understand the user's intent.
2. Add fine details to the prompt to make it more specific and detailed.
3. Add styling, colours, fonts, layouts, etc. to the prompt to more aesthetic and visually appealing output.
4. Return the rewritten prompt in the json format.

Prompt:
{instruction}

Output:
{output}
"""

    new_dataset.append(prompt)

with open('dataset.json', 'w') as f:
    json.dump(new_dataset, f)