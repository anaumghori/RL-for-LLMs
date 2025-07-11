# GRPO Training System
A distributed training system for fine-tuning Qwen model using Group-wise Relative Preference Optimization (GRPO) on the GSM8K dataset. The model is trained to generate R1-style reasoning with structured think/answer format while maximizing correctness rewards.

## Key Features
- **Distributed Training:** Uses DeepSpeed for efficient multi-GPU training   
- **Reward-based Learning:** Combines correctness rewards and format compliance rewards   
- **KL Regularization:** Prevents the model from deviating too far from the reference policy   
- **PPO Clipping:** Implements Proximal Policy Optimization for stable training   
- **Answer Verification:** Uses symbolic parsing to verify GSM8K answer correctness   

## Architecture
```
├── GRPO/  
│   ├── main_grpo.py            # Handles model training, data generation, and loss computation
│   ├── server.py               # Provides reference model log probabilities for KL divergence computation  
│   ├── configs.py              # Contains hyperparameters and model setup
|   ├── step_X/                 # Model checkpoints (generated during training)
```
## Usage
Starting the Reference Server. The server runs on localhost:59875 and provides reference model log probabilities.
```
python server.py
```
To run only data generation (useful for testing):
```
python main_grpo.py genonly
```
Full Training: This starts the complete training pipeline with distributed training support.
```
deepspeed main_grpo.py
```
## Results
The files do not contain code for visualizing results, but you can generate visualizations by modifying the code to use wandb or matplotlib, or by saving logs and generating plots separately.    


<img width="500" height="500" alt="GRPO_LOSS" src="https://github.com/user-attachments/assets/d3db0c1e-5fa8-41e1-bc4f-76915ee2690d" />


<img width="500" height="500" alt="GRPO_FORMAT" src="https://github.com/user-attachments/assets/606592cc-8ef5-4e22-8a82-a080aa012b7c" />



