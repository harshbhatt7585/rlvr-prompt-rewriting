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
from datasets import load_dataset
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


response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        },
        {
            "role": "assistant",
            "content": "Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the worlds largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, its no wonder that Paris is one of the most popular tourist destinations in the world.",
        },
        {
            "role": "user",
            "content": "What is so great about #1?",
        }
    ],
    max_completion_tokens=13107,
    temperature=1.0,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    model=deployment

)

print(response.choices[0].message.content)