import json, re, random, requests, time, torch, sys
import torch.distributed as dist
import deepspeed
from tqdm import tqdm
from math_verify import parse, verify, ExprExtractionConfig
from server import bytes_list_to_list, bytes_to_tensor, tensor_to_bytes, make_bytes_list
from configs import *

# Training batch data containing inputs, rewards, and reference log probabilities
def get_training_batch_from_server():
    try:
        response = requests.get(f"{REFERENCE_SERVER_URL}/get").content
        if response == b'empty':
            return None
    except:
        return None
    
    # Deserialize the batch data
    data_list = bytes_list_to_list(response)
    batch_data = json.loads(data_list[0])
    batch_data['inputs'] = bytes_to_tensor(data_list[1])      # Tokenized prompt + generation
    batch_data['rewards'] = bytes_to_tensor(data_list[2])     # Reward for each token
    batch_data['ref_logps'] = bytes_to_tensor(data_list[3])   # Reference model log-probabilities
    
    # Include generation log-probabilities if available
    if len(data_list) == 5:
        batch_data['gen_logps'] = bytes_to_tensor(data_list[4])
    
    return batch_data

def format_prompts(user_prompts):
    """
    Args: user_prompts (list): List of user prompt strings
    Returns: list: List of formatted prompts ready for tokenization
    """
    formatted_prompts = []
    for user_prompt in user_prompts:
        formatted_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)
    
    return formatted_prompts

def generate_answers(prompts):
    """
    Generate answers for a list of prompts using the current model.
    Args: prompts (list): List of prompt strings 
    Returns: list: List of generated answers, empty if prompt too long
    """
    formatted_prompts = format_prompts(prompts)
    tokenized_inputs = tokenizer(formatted_prompts, return_tensors="pt",
        padding=True, padding_side="left", add_special_tokens=False)
    
    prompt_length = tokenized_inputs["input_ids"].shape[-1]
    if prompt_length > MAX_PROMPT_LENGTH:
        return []
    
    tokenized_inputs = {k: v.to(generation_model.device) for k, v in tokenized_inputs.items()}
    with torch.inference_mode():
        output_ids = generation_model.generate(**tokenized_inputs, generation_config=GENERATION_CONFIG)
    
    # Extract only the generated tokens (remove prompt part)
    generated_token_ids = output_ids[:, prompt_length:]
    generated_answers = [
        tokenizer.decode(token_ids, skip_special_tokens=True).replace('<|endoftext|>', '')
        for token_ids in generated_token_ids
    ]
    
    return generated_answers

def calculate_correctness_reward(question_answer_item, generated_answer):
    """
    Args: question_answer_item (dict): Contains 'Q' (question) and 'A' (correct answer)
          generated_answer (str): The model's generated answer
    Returns: float: Reward value (1.0 for correct, -1.0 for incorrect)
    """
    # Extract numerical values from the answer
    number_pattern = r'\d+\.\d+|\d+/\d+|\d+'
    found_numbers = re.findall(number_pattern, generated_answer)
    
    if len(found_numbers) == 0:
        return -1.0
    
    # Use the last number found as the final answer
    predicted_answer = found_numbers[-1]
    
    # Parse and verify the answer
    parsed_prediction = parse(predicted_answer, extraction_config=[ExprExtractionConfig()])
    parsed_ground_truth = parse(question_answer_item["A"], extraction_config=[ExprExtractionConfig()])
    
    return 1.0 if verify(parsed_prediction, parsed_ground_truth) else -1.0

def calculate_format_reward(question_answer_item, generated_answer):
    """
    Calculate reward based on whether the answer follows the required format.
    
    Args: question_answer_item (dict): Contains 'Q' (question) and 'A' (correct answer)
           generated_answer (str): The model's generated answer
    Returns: float: Reward value (1.25 for correct format, -1.0 for incorrect format)
    """
    # Check if answer follows the <think>...</think><answer>...</answer> format
    format_pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    is_correct_format = re.match(format_pattern, generated_answer, re.DOTALL | re.VERBOSE)
    
    return 1.25 if is_correct_format else -1.0

def compute_log_probabilities(logits, input_token_ids):
    """
    Compute log probabilities for each token given the model logits.
    
    Args: logits (torch.Tensor): Model logits of shape (batch_size, seq_len, vocab_size)
          input_token_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len)  
    Returns: torch.Tensor: Per-token log probabilities of shape (batch_size, seq_len)
    """
    per_token_log_probs = []
    
    # Process each sequence separately to reduce memory usage
    for logits_sequence, token_ids_sequence in zip(logits, input_token_ids):
        log_probabilities = logits_sequence.log_softmax(dim=-1)
        token_log_prob = torch.gather(
            log_probabilities, 
            dim=1, 
            index=token_ids_sequence.unsqueeze(1)
        ).squeeze(1)
        per_token_log_probs.append(token_log_prob)
    
    return torch.stack(per_token_log_probs)

def prepare_training_data(question_answer_items):
    """
    Args: question_answer_items (list): List of dictionaries with 'Q' and 'A' keys   
    Returns: tuple: (prompt_input_ids, output_token_ids, rewards, generated_answers) or (None, None, None, None)
    """
    # Extract questions and generate answers
    questions = [item["Q"] for item in question_answer_items]
    generated_answers = generate_answers(questions)
    
    if not generated_answers:
        return None, None, None, None
    
    # Calculate rewards for each generated answer
    all_rewards = []
    for i, question_answer_item in enumerate(question_answer_items):
        # Each question generates 3 answers
        start_idx = i * 3
        end_idx = (i + 1) * 3
        
        for answer in generated_answers[start_idx:end_idx]:
            correctness_reward = calculate_correctness_reward(question_answer_item, answer)
            format_reward = calculate_format_reward(question_answer_item, answer)
            total_reward = correctness_reward + format_reward
            all_rewards.append(total_reward)
    
    # Prepare formatted prompts and tokenize them
    formatted_prompts = format_prompts(questions)
    prompt_token_ids = tokenizer(formatted_prompts, return_tensors="pt", padding=True,
        padding_side="left", add_special_tokens=False)["input_ids"]
    
    # Tokenize generated answers
    output_token_ids = tokenizer(generated_answers, return_tensors="pt", padding=True,
        padding_side="right", add_special_tokens=False)["input_ids"]
    
    return prompt_token_ids, output_token_ids, torch.tensor(all_rewards, dtype=torch.float32), generated_answers

# Upload a training batch to the reference server.
def server_upload(batch_data):
    serialized_batch = make_bytes_list(batch_data)
    requests.post(f"{REFERENCE_SERVER_URL}/upload", data=serialized_batch)


def run_generation_mode(num_iterations=10, process_rank=0):
    """
    Args: num_iterations (int): Number of generation iterations to run
          process_rank (int): Rank of the current process for distributed training
    """
    start_time = time.time()
    
    for iteration in range(num_iterations):
        sampled_questions = random.sample(question_answer_pairs, QUESTIONS_PER_BATCH) # Sample random question-answer pairs
        prompt_ids, output_ids, rewards, answers = prepare_training_data(sampled_questions) # Generate training data
        if prompt_ids is None:
            continue
        
        # Skip if rewards don't vary enough (no learning signal)
        if (rewards.max() - rewards.min()).item() < 0.01:
            continue
        normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        
        # Prepare data for server upload
        num_sequences_per_prompt = output_ids.shape[0] // prompt_ids.shape[0]
        prompt_length = prompt_ids.shape[1]
        repeated_prompts = prompt_ids.repeat(1, num_sequences_per_prompt).view(-1, prompt_length) # Repeat prompts to match output sequences
        full_sequences = torch.cat([repeated_prompts, output_ids], dim=1) # Concatenate prompts and outputs
        
        # Prepare batch data for upload
        batch_data = [
            json.dumps({"plen": prompt_length}).encode(),
            tensor_to_bytes(full_sequences),
            tensor_to_bytes(normalized_rewards)
        ]
        
        # Add generation log probabilities if required
        if COMPUTE_GENERATION_LOGPROBS:
            with torch.inference_mode():
                sequences_on_device = full_sequences.to(generation_model.device)
                model_logits = generation_model(sequences_on_device).logits
                generation_log_probs = compute_log_probabilities(
                    model_logits[:, :-1, :], 
                    sequences_on_device[:, 1:]
                )
                # Extract only the generation part (exclude prompt)
                generation_only_log_probs = generation_log_probs[:, prompt_length-1:]
                
            batch_data.append(tensor_to_bytes(generation_only_log_probs.cpu()))
        
        # Upload batch to server
        server_upload(batch_data)
    elapsed_time = time.time() - start_time

def compute_grpo_loss(batch_data, engine):
    """
    Args: batch_data (dict): Batch containing inputs, rewards, and reference log probabilities
          engine: DeepSpeed engine
    Returns: torch.Tensor: Computed loss value
    """
    prompt_length = batch_data['plen']
    input_sequences = batch_data['inputs'].to(engine.device)
    advantages = batch_data['rewards'].to(engine.device).unsqueeze(1)  # Normalized rewards
    
    # Forward pass through the model
    model_logits = engine(input_sequences).logits
    model_logits = model_logits[:, :-1, :]  # Exclude last logit (no corresponding target)
    target_token_ids = input_sequences[:, 1:]  # Exclude first token (no corresponding logit)
    
    # Compute per-token log probabilities
    current_log_probs = compute_log_probabilities(model_logits, target_token_ids)
    # Focus only on the generation part (exclude prompt)
    generation_log_probs = current_log_probs[:, prompt_length-1:]
    reference_log_probs = batch_data['ref_logps'].to(generation_log_probs.device)
    
    # Compute KL divergence penalty
    kl_divergence = torch.exp(reference_log_probs - generation_log_probs) - (reference_log_probs - generation_log_probs) - 1
    
    # Create mask for actual tokens (not padding)
    completion_mask = (input_sequences[:, prompt_length:] != tokenizer.pad_token_id).int()
    
    # Compute policy loss
    if 'gen_logps' in batch_data:
        importance_ratio = torch.exp(generation_log_probs - batch_data['gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(importance_ratio, 1 - PPO_CLIP_PARAMETER, 1 + PPO_CLIP_PARAMETER)
        policy_loss = torch.min(importance_ratio * advantages, clipped_ratio * advantages)
    else:
        # Simple policy gradient (no clipping)
        policy_loss = torch.exp(generation_log_probs - generation_log_probs.detach()) * advantages
        assert COMPUTE_GENERATION_LOGPROBS is False
    
    # Combine policy loss with KL penalty
    per_token_loss = -(policy_loss - KL_PENALTY_COEFFICIENT * kl_divergence)
    
    # Average loss over valid tokens
    sequence_losses = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
    final_loss = sequence_losses.mean()
    
    return final_loss

def save_model_checkpoint(step_number, engine):
    dist.barrier()  # Synchronize all processes
    
    if torch.distributed.get_rank() == 0:
        checkpoint_dir = f"./step_{step_number}"
        
        # Get model state dict and move to CPU
        model_state_dict = engine.module.state_dict()
        cpu_state_dict = type(model_state_dict)({
            key: value.cpu() for key, value in model_state_dict.items()
        })
        
        # Save model and tokenizer
        engine.module.save_pretrained(checkpoint_dir, state_dict=cpu_state_dict)
        tokenizer.save_pretrained(checkpoint_dir)
    
    dist.barrier()

if __name__ == '__main__':  
    # Handle generation-only mode
    if 'genonly' in sys.argv:
        generation_model = model.to('cuda')
        run_generation_mode(num_iterations=999999)
        sys.exit()
    
    # Initialize DeepSpeed
    engine, optimizer, _, _ = deepspeed.initialize(config=DEEPSPEED_CONFIG, model=model, model_parameters=model.parameters())
    generation_model = engine
    
    # Run initial generation phase
    current_rank = torch.distributed.get_rank()
    run_generation_mode(num_iterations=10, process_rank=current_rank)
    
    # Main training loop
    training_steps = range(1, TOTAL_TRAINING_STEPS + 1)
    if current_rank == 0:
        training_steps = tqdm(training_steps, desc="Training Progress")
    
    for step in training_steps:
        training_batch = get_training_batch_from_server()
        
        # Generate more data if no batch available
        while training_batch is None:
            run_generation_mode(num_iterations=10, process_rank=current_rank)
            training_batch = get_training_batch_from_server()
        
        loss = compute_grpo_loss(training_batch, engine)
        engine.backward(loss)
        engine.step()
        
        if current_rank == 0:
            training_steps.set_description(f"Training Progress - Loss: {loss.item():.6f}")
        
        if step % MODEL_SAVE_INTERVAL == 0:
            save_model_checkpoint(step, engine)