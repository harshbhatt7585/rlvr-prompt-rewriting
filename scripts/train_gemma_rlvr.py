"""
Training a gemma model to rewrite prompts to generate best outputs from model.

"Autoregressive Prompt Booster for Coding Models"

Approch:
1. Use gemma 3 1b model to rewrite prompts for himself to generate best outputs for a given task.
2. Judge the output with openai gpt4.1 model to score between 0 to 5. This act a reward.
3. Use the reward to update the model.
4. Use LoRA to finetune the model.

You should train a model that rewrites vague frontend prompts into high-quality UI design prompts.

This is far more useful than coding prompt rewriting because:

Frontend design is extremely subjective

Users struggle to express what they want

Designers struggle to translate vague instructions

LLMs can do UI, but only when the prompt is detailed

Your Gemma 1B learns to turn:

“Make this page look nicer”

into:

“Create a modern, minimal, responsive landing page with a bold title, a two-column layout, generous white space, a soft gradient header, and Tailwind CSS. Use rounded cards, a clean hero section, and consistent spacing with 8-point grid.”

"""

import os
import torch
import transformers
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)


model_name = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.15,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

deployment = "gpt-4"


def load_dataset():
    dataset = load_from_disk("./dataset")
    return dataset['train']

def judge_response(answer):

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are a coding judge. You are given html code and your task is to judge the code based on the:
                The deisgn is asthetic and visually appealing. Give a score between 0 to 5. 0 is the worst and 5 is the best.
                """
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_completion_tokens=13107,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=deployment

    )

    return response.choices[0].message.content


def generate_response(prompt):
    messages = apply_chat_template(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a prompt enhancement assistant. You are given a prompt and you need to enhance it to add fine details for world class output.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    inputs = tokenizer(messages, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=1024)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
    


dataset = load_dataset()
EPOCHS = 10
for epoch in range(EPOCHS):
    for data in dataset:
        prompt = data['rewriting_prompt']
        response = generate_response(prompt)
        print(response)
        break