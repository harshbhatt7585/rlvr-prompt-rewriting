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
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import AzureOpenAI
from dotenv import load_dotenv
import json
import random
import numpy as np
import wandb

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

device = 'cuda'
model_name = "google/gemma-3-1b-it"

# Configure 8-bit quantization for better speed/quality balance on 16GB GPU
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",  # Use SDPA (built into PyTorch 2.0+, no flash-attn needed)
)


lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.15,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Optional: torch.compile for additional speedup (may not work with quantization + PEFT)
# Uncomment if not using quantization or if it works with your setup
# model = torch.compile(model, mode="reduce-overhead")

deployment = "gpt-4"

# PPO Hyperparameters (Optimized for 16GB GPU)
LEARNING_RATE = 1e-5
BATCH_SIZE = 8  # Increased from 4 to 8 for better GPU utilization
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 8 * 4 = 32
BUFFER_SIZE = 256  # Increased buffer size for more diverse experiences
CLIP_EPSILON = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
GAMMA = 0.99
MIN_BUFFER_SIZE = 32  # Increased minimum buffer size


def get_dataset():
    dataset = load_from_disk("./dataset")
    return dataset['train']

def judge_response(answer):

    response = client.chat.completions.create(
        model=deployment,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """You are an HTML/Frontend judge.

You will receive raw HTML code. Your job is to judge whether the design is:

- visually appealing
- modern
- readable
- well-structured
- aesthetically pleasing

Score the design from **0 to 5**.

You MUST output only a JSON object with this exact schema:

{
  "score": <number>
}

No explanations. No extra text."""
            },
            {
                "role": "user",
                "content": answer
            }
        ],
        temperature=0.01,
        top_p=1.0
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
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # Optimized generation settings for speed
    with torch.cuda.amp.autocast():  # Use mixed precision for generation too
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # Reduced from 1024 for faster generation
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache for faster generation
        )

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
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,  # Greedy decoding for more deterministic code generation
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return answer


def compute_log_probs(prompts, responses, requires_grad=False):
    """Compute log probabilities for generated responses given prompts (BATCHED).

    Args:
        prompts: List of input prompts
        responses: List of generated responses
        requires_grad: If True, compute gradients (for training). If False, use no_grad (for storing old probs)
    """
    # Format all prompts
    formatted_prompts = []
    full_texts = []
    for prompt, response in zip(prompts, responses):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(formatted)
        full_texts.append(formatted + response)

    # Batch tokenize with padding
    prompt_inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    full_inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Get model outputs in batch
    if requires_grad:
        outputs = model(**full_inputs)
        logits = outputs.logits
    else:
        with torch.no_grad():
            outputs = model(**full_inputs)
            logits = outputs.logits

    # Compute log probs for each sequence in batch
    log_probs_list = []
    for i, (prompt_len, full_len) in enumerate(zip(prompt_inputs.attention_mask.sum(dim=1), full_inputs.attention_mask.sum(dim=1))):
        # Get the response portion
        response_start = prompt_len.item()
        response_end = full_len.item()

        if response_end > response_start:
            # Get logits and tokens for response
            response_logits = logits[i, response_start-1:response_end-1, :]
            response_tokens = full_inputs.input_ids[i, response_start:response_end]

            # Compute log probabilities
            log_probs = F.log_softmax(response_logits, dim=-1)
            token_log_probs = log_probs[range(len(response_tokens)), response_tokens]

            # Average log prob across tokens
            mean_log_prob = token_log_probs.mean()
            log_probs_list.append(mean_log_prob)
        else:
            # Handle edge case where response is empty
            log_probs_list.append(torch.tensor(0.0, device=model.device))

    return torch.stack(log_probs_list)


def compute_advantage(rewards):
    """Compute normalized advantages from rewards."""
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

    # Normalize rewards to have mean 0 and std 1
    if len(rewards) > 1:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    else:
        advantages = rewards

    return advantages


def compute_policy_loss(old_log_probs, new_log_probs, advantages, clip_epsilon=CLIP_EPSILON):
    """Compute PPO clipped policy loss."""
    ratio = torch.exp(new_log_probs - old_log_probs)

    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()

    return policy_loss


def compute_ppo_loss(prompts, responses, rewards, old_log_probs):
    """Compute complete PPO loss."""
    # Get current policy log probs (with gradients for training)
    new_log_probs = compute_log_probs(prompts, responses, requires_grad=True)

    # Compute advantages
    advantages = compute_advantage(rewards)

    # Compute policy loss with PPO clipping
    policy_loss = compute_policy_loss(old_log_probs, new_log_probs, advantages)

    return policy_loss, new_log_probs


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        prompts, enhanced_prompts, scores, old_log_probs = zip(*batch)
        return list(prompts), list(enhanced_prompts), list(scores), list(old_log_probs)

    def __len__(self):
        return len(self.buffer)




def generate_code_with_gpt(prompt):
    response = client.chat.completions.create(
        messages = [
        {
            "role": "system",
            "content": "You are a code generator. You are given a prompt and you need to generate the code based on the prompt.",
        },
        {
            "role": "user",
            "content": prompt + " Only return the code in the json format. Do not provide any other text.",
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


# Initialize replay buffer
replay_buffer = ReplayBuffer(max_size=BUFFER_SIZE)

# Initialize optimizer (only trainable parameters)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Initialize gradient scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Create checkpoints directory
os.makedirs("./checkpoints", exist_ok=True)

dataset = get_dataset()
EPOCHS = 10

# Initialize wandb
wandb.init(
    project="gemma-prompt-rewriting-ppo",
    config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "buffer_size": BUFFER_SIZE,
        "clip_epsilon": CLIP_EPSILON,
        "value_coef": VALUE_COEF,
        "entropy_coef": ENTROPY_COEF,
        "gamma": GAMMA,
        "min_buffer_size": MIN_BUFFER_SIZE,
        "epochs": EPOCHS,
        "model": model_name,
        "quantization": "8bit",
        "mixed_precision": True,
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.15,
    }
)

print(f"Starting PPO training for {EPOCHS} epochs...")
print(f"Batch size: {BATCH_SIZE}, Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}, Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Buffer size: {BUFFER_SIZE}, Min buffer size: {MIN_BUFFER_SIZE}")
print(f"Using 8-bit quantization and mixed precision training for optimal 16GB GPU utilization")

global_step = 0
episode_rewards = []  # Track rewards for running average
accumulation_step = 0  # Track gradient accumulation steps

for epoch in range(EPOCHS):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"{'='*50}")

    for idx, data in enumerate(dataset):
        prompt = data['rewriting_prompt']

        # Generate enhanced prompt using current policy
        enhanced_prompt = generate_response(prompt)
        print(f"\n[Step {global_step}] Original prompt: {prompt[:100]}...")
        print(f"Enhanced prompt: {enhanced_prompt[:150]}...")

        # Generate code using GPT with enhanced prompt
        try:
            generated_code = generate_code_with_gpt(enhanced_prompt)
            print(f"Generated code (first 100 chars): {generated_code[:100]}...")

            # Judge the generated code
            score_response = judge_response(generated_code)
            score_data = json.loads(score_response.strip())
            reward = score_data['score']
            print(f"Reward: {reward}/5")

            # Compute old log probs for the generated response
            old_log_probs = compute_log_probs([prompt], [enhanced_prompt])

            # Store experience in replay buffer
            experience = (prompt, enhanced_prompt, reward, old_log_probs[0].item())
            replay_buffer.add(experience)
            episode_rewards.append(reward)
            print(f"Buffer size: {len(replay_buffer)}/{BUFFER_SIZE}")

            # Compute running averages
            avg_reward_last_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            avg_reward_last_50 = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)

            # Log to wandb
            wandb.log({
                "reward": reward,
                "avg_reward_last_10": avg_reward_last_10,
                "avg_reward_last_50": avg_reward_last_50,
                "buffer_size": len(replay_buffer),
                "old_log_prob": old_log_probs[0].item(),
                "step": global_step,
                "epoch": epoch + 1,
            })

        except Exception as e:
            print(f"Error during generation/evaluation: {e}")
            continue

        # Start training once we have enough experiences
        if len(replay_buffer) >= MIN_BUFFER_SIZE:
            print(f"\n[Training] Starting PPO update (accumulation step {accumulation_step + 1}/{GRADIENT_ACCUMULATION_STEPS})...")

            # Sample batch from replay buffer
            prompts, enhanced_prompts, rewards, old_log_probs_batch = replay_buffer.sample(BATCH_SIZE)

            # Convert old_log_probs to tensor
            old_log_probs_tensor = torch.tensor(old_log_probs_batch, device=device)

            try:
                # Compute PPO loss with mixed precision
                model.train()

                with torch.cuda.amp.autocast():
                    policy_loss, new_log_probs = compute_ppo_loss(
                        prompts, enhanced_prompts, rewards, old_log_probs_tensor
                    )
                    # Scale loss by accumulation steps
                    policy_loss = policy_loss / GRADIENT_ACCUMULATION_STEPS

                # Backpropagation with gradient scaling
                scaler.scale(policy_loss).backward()

                accumulation_step += 1

                # Update weights after accumulating gradients
                if accumulation_step % GRADIENT_ACCUMULATION_STEPS == 0:
                    # Unscale gradients and clip
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    print(f"Policy loss: {policy_loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f} (accumulated)")

                    # Log training metrics to wandb
                    wandb.log({
                        "train/policy_loss": policy_loss.item() * GRADIENT_ACCUMULATION_STEPS,
                        "train/avg_batch_reward": np.mean(rewards),
                        "train/min_batch_reward": np.min(rewards),
                        "train/max_batch_reward": np.max(rewards),
                        "train/step": global_step,
                        "train/accumulation_step": accumulation_step,
                    })
                else:
                    print(f"Gradient accumulated ({accumulation_step % GRADIENT_ACCUMULATION_STEPS}/{GRADIENT_ACCUMULATION_STEPS})")

            except Exception as e:
                print(f"Error during training step: {e}")
                import traceback
                traceback.print_exc()
                continue

        global_step += 1

        # Save checkpoint every 50 steps
        if global_step % 50 == 0 and global_step > 0:
            checkpoint_path = f"./checkpoints/gemma_ppo_step_{global_step}"
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"\nCheckpoint saved to {checkpoint_path}")

            # Log checkpoint save to wandb
            wandb.log({"checkpoint_step": global_step})

# Save final model
final_model_path = "./checkpoints/gemma_ppo_final"
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"\nFinal model saved to {final_model_path}")

# Log final model to wandb
wandb.save(f"{final_model_path}/*")
print("\nTraining completed!")

# Finish wandb run
wandb.finish()