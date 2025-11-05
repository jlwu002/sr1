import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
import os
import json
from qa_em import compute_score_subem, compute_score_em, calculate_f1

def get_prompt_with_chat_template(tokenizer, prompt):
    return tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)

def log_to_file(*args, fname=None):
    with open(fname, "a") as f:
        print(*args, file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--file_name', type=str, required=True)

    args = parser.parse_args()

    dataset = load_from_disk(args.data_path)
    
    log_to_file(args.model_path, fname=args.file_name)

    model = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=2, ###change to 4
        gpu_memory_utilization=0.9,
        max_model_len=32768,
    )

    sampling_params = SamplingParams(
        temperature=0.,
        max_tokens=8192,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    dataset_names = set(dataset['data_source'])
    model_name = args.model_path.split('/')[-1]

    for dataset_name in dataset_names:
        log_to_file(f'Processing {dataset_name}...', fname=args.file_name)
        current_dataset = dataset.filter(lambda example: example['data_source'] == dataset_name)
        prompts = [get_prompt_with_chat_template(tokenizer, raw_prompt) for raw_prompt in current_dataset['prompt']]
        ground_truths = [x['ground_truth'] for x in current_dataset['reward_model']]
        outputs = model.generate(prompts, sampling_params)
        outputs = [output.outputs[0].text for output in outputs]
        save_path = os.path.join('results/', model_name, dataset_name)
        os.makedirs(save_path, exist_ok=True)
        
        with open(os.path.join(save_path, args.output_path), 'w') as f:
            for output, ground_truth in zip(outputs, ground_truths):
                f.write(json.dumps({'output': output, 'ground_truth': ground_truth}) + '\n')
        sub_em_scores = []
        em_scores = []
        gold_answers = [x['target'] for x in ground_truths]
        f1_score = calculate_f1(gold_answers, outputs)
        for output, ground_truth in zip(outputs, ground_truths):
            sub_em_score = compute_score_subem(output, ground_truth)
            sub_em_scores.append(sub_em_score)
            em_score = compute_score_em(output, ground_truth)
            em_scores.append(em_score)
        log_to_file(f'{dataset_name} Average sub_em score: {sum(sub_em_scores) / len(sub_em_scores)}', fname=args.file_name)
        log_to_file(f'{dataset_name} Average em score: {sum(em_scores) / len(em_scores)}', fname=args.file_name)
        log_to_file(f'{dataset_name} Average f1 score: {f1_score}', fname=args.file_name)