# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the QA dataset to parquet format
"""

import re
import os
import json
import random
import datasets
from pathlib import Path

import argparse

from prompt_template_sr1 import PROMPT_SR1

random.seed(42)


def make_prefix(dp, args):
    question = dp['question']
    context = dp['context']
    prefix = PROMPT_SR1.format_map({"question": question, "context": context})
    return prefix

def make_context(dp, data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'bamboogle', 'musique']:
        contents = dp['context']
        titles = [item['title'] for item in contents]
        contents = [item['text'] for item in contents]
        context = ""
        idx = 1
        for title, content in zip(titles, contents):
            context += f'Doc {idx} (Title: {title})\n{content}\n\n'
            idx += 1
            if idx > 10:
                break
    elif data_source == '2wikimultihopqa':
        content = dp['metadata']['context']
        titles = content['title']
        contents = content['content']
        context = ""
        idx = 1
        for title, content in zip(titles, contents):
            curr_content = ' '.join(content)
            context += f'Doc {idx} (Title: {title})\n{curr_content}\n\n'
            idx += 1
            if idx > 10:
                break
    elif data_source == 'hotpotqa':
        content = dp['metadata']['context']
        titles = content['title']
        contents = content['sentences']
        context = ""
        idx = 1
        for title, content in zip(titles, contents):
            curr_content = ' '.join(content)
            context += f'Doc {idx} (Title: {title})\n{curr_content}\n\n'
            idx += 1
            if idx > 10:
                break
    else:
        raise NotImplementedError
    return context


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_sources', default='nq,hotpotqa,triviaqa,popqa,2wikimultihopqa,musique,bamboogle')
    parser.add_argument('--all_data', action='store_true')

    args = parser.parse_args()
    data_sources = args.data_sources.split(',')
    all_dataset = []


    for data_source in data_sources:
        if data_source == 'nq':
            if args.all_data:
                with open('your_path/data/nq/nq_test_all_retrieved.jsonl', 'r') as f:
                    data = [json.loads(line) for line in f]
                test_dataset = datasets.Dataset.from_list(data)
            else:
                with open('your_path/data/nq/nq_test_retrieved.jsonl', 'r') as f:
                    data = [json.loads(line) for line in f]
                test_dataset = datasets.Dataset.from_list(data)
        elif data_source in ['triviaqa', 'popqa', 'bamboogle', 'musique']:
            with open(f'your_path/data/{data_source}/{data_source}_test_all_retrieved.jsonl', 'r') as f:
                data = [json.loads(line) for line in f]
            test_dataset = datasets.Dataset.from_list(data)
        else:
            dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)
            if 'test' in dataset:
                print(f'Using the {data_source} test dataset...')
                test_dataset = dataset['test']
            elif 'dev' in dataset:
                print(f'Using the {data_source} dev dataset...')
                test_dataset = dataset['dev']
            else:
                print(f'Using the {data_source} train dataset...')
                test_dataset = dataset['train']
            if not args.all_data:
                # random sample
                test_dataset = test_dataset.select(random.sample(list(range(len(test_dataset))), 500))

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                example['context'] = make_context(example, data_source)
                example['question'] = example['question'].strip()
                if example['question'][-1] != '?':
                    example['question'] += '?'
                question = make_prefix(example, args=args)
                solution = {
                    "target": example['golden_answers'],
                }

                data = {
                    "data_source": data_source,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "general-qa",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
                return data

            return process_fn

        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True).remove_columns(test_dataset.column_names)
        print(f'{data_source} test_dataset: {test_dataset[0]}')
        all_dataset.append(test_dataset)

    all_test_dataset = datasets.concatenate_datasets(all_dataset)
    all_test_dataset.save_to_disk("multihop_test_dataset")
