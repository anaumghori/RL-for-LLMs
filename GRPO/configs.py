from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset
import os
import torch

# Training hyperparameters
QUESTIONS_PER_BATCH = 1            # Number of episodes processed per PPO batch 
MAX_PROMPT_LENGTH = 400           
TOTAL_TRAINING_STEPS = 100       
MODEL_SAVE_INTERVAL = 20        
KL_PENALTY_COEFFICIENT = 0.04      # KL penalty coefficient to keep policy close to reference model
PPO_CLIP_PARAMETER = 0.2           # PPO clipping parameter (epsilon)
COMPUTE_GENERATION_LOGPROBS = True # Whether to compute log probabilities during generation
MODEL_PATH = "Qwen/Qwen3-8B"
REFERENCE_SERVER_URL = "http://localhost:59875"

# DeepSpeed configuration
DEEPSPEED_CONFIG = {
    "train_micro_batch_size_per_gpu": QUESTIONS_PER_BATCH * 3,
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-6}
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}

SYSTEM_PROMPT = """
You are a helpful, intelligent assistant.
When the user asks a question, you carefully reason through the answer step by step, and then give the final answer.
Your response must follow this structure:
<think> Your detailed reasoning process goes here. </think><answer> Your final answer goes here. </answer>
Respond only with the <think> and <answer> tags, and do not include any extra explanations outside them.
"""

# Model and Dataset
os.environ['TOKENIZERS_PARALLELISM'] = 'true' # Set tokenizer parallelism for better performance

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16,
    _attn_implementation="sdpa"
)

# Generation configuration
GENERATION_CONFIG = GenerationConfig(
    max_new_tokens=512,
    do_sample=True,
    temperature=0.9,
    num_return_sequences=3,  # Generate 3 responses per prompt
    pad_token_id=tokenizer.pad_token_id
)

dataset = load_dataset("openai/gsm8k", "main", split="train")
question_answer_pairs = []
for question, answer in zip(dataset['question'], dataset['answer']):
    final_answer = answer.split('####')[-1].strip()
    question_answer_pairs.append({'Q': question, 'A': final_answer})