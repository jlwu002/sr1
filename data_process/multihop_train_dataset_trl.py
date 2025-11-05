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
from datasets import Dataset
from pathlib import Path

#from verl.utils.hdfs_io import copy, makedirs
import argparse

from prompt_template_sr1 import PROMPT_SR1

random.seed(42)


def make_prefix(dp, args):
    question = dp['question']
    context = dp['context']
    prefix = PROMPT_SR1.format_map({"question": question, "context": context})
    return prefix

def make_context(dp, data_source):
    if data_source == '2wikimultihopqa':
        content = dp['metadata']['context']
        titles = content['title']
        contents = content['content']
        context = ""
        idx = 1
        for title, content in zip(titles, contents):
            curr_content = ' '.join(content)
            context += f'Doc {idx} (Title: {title})\n{curr_content}\n\n'
            idx += 1
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
    else:
        raise NotImplementedError
    return context


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_sources', default='2wikimultihopqa,hotpotqa')

    args = parser.parse_args()
    data_sources = args.data_sources.split(',')
    all_dataset = []

    for data_source in data_sources:
        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)
        train_dataset = dataset['train']
        # random sample
        train_dataset = train_dataset.select(random.sample(list(range(len(train_dataset))), 7000))

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                example['context'] = make_context(example, data_source)
                example['question'] = example['question'].strip()
                if example['question'][-1] != '?':
                    example['question'] += '?'
                question = make_prefix(example, args=args)

                data = {
                    "prompt": question,
                    "ground_truths": example['golden_answers'],
                    "id": example['id'],
                }
                return data

            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True).remove_columns(train_dataset.column_names)
        print(f'{data_source} train_dataset: {train_dataset[0]}')
        all_dataset.append(train_dataset)

    all_train_dataset = datasets.concatenate_datasets(all_dataset)
    all_train_dataset.save_to_disk("multihop_train_dataset")
