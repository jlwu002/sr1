from datasets import load_dataset, load_from_disk
from trl import GRPOConfig, GRPOTrainer
from qa_em import compute_score_em
# Load dataset
dataset = load_from_disk("data_process/multihop_train_dataset")
# dataset_test = load_from_disk("/opt/dlami/nvme/sr1_trl/Structure-R1-main/data_process/multihop_test_dataset")
# Define the reward function
def reward_compute_func(prompts, completions, context, ground_truths, **kwargs):
    return [compute_score_em(comp, gt) for comp, gt in zip(completions,ground_truths)]

# Define training arguments
training_args = GRPOConfig(
    output_dir="results/grpo_qwen_7b_sr1_warmup",
    logging_steps=1,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_tensor_parallel_size=4,
    vllm_gpu_memory_utilization=0.2,
    max_prompt_length=6000,
    max_completion_length=4096,
    max_steps=800, 
    num_generations=8,
    generation_batch_size=512,
    per_device_train_batch_size=4,
    push_to_hub=False,
    save_steps=200,
)

# Create and run the trainer
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-7B-Instruct",
    reward_funcs=reward_compute_func,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
