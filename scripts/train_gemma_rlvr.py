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
import json

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

device = 'cuda'
model_name = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.15,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

deployment = "gpt-4"


def get_dataset():
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
                "content": answer,
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
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model.generate(**inputs, max_new_tokens=1024)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return answer
    

def generate_code(prompt):
    messages = [
        {
            "role": "system",
            "content": "You are a code generator. You are given a prompt and you need to generate the code based on the prompt.",
        },
        {
            "role": "user",
            "content": prompt + " Only return the code in the json format. Do not provide any other text.",
        },
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.01)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return answer


dataset = get_dataset()
EPOCHS = 10
for epoch in range(EPOCHS):
    for data in dataset:
        prompt = data['rewriting_prompt']
        enhanced_prompt = generate_response(prompt)
        print(enhanced_prompt)
        generated_code = generate_code(enhanced_prompt)
        print(generated_code)
        break
    break