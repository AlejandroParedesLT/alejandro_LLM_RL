
import re
from datasets import load_dataset, Dataset

def getCodingDataset(partition: str, tokenizer=None) -> Dataset:
    # Load and prep dataset
    SYSTEM_PROMPT = """
    Respond in the following format:
    <think>
    ...
    </think>
    <answer>
    ...
    </answer>
    """

    XML_COT_FORMAT = """\
    <think>
    {think}
    </think>
    <answer>
    {answer}
    </answer>
    """

    def extract_hash_answer(text: str) -> str | None:
        if "</think>" not in text:
            #raise ValueError("No <think> tag found in the text.")
            return text[-500:] 
        return text.split("</think>")[1].strip()

    # uncomment middle messages for 1-shot prompting
    def get_coding_nt(split = "train") -> Dataset:
        data = load_dataset('open-r1/codeforces-cots', 'solutions')[split] # type: ignore
        data = data.remove_columns([col for col in data.column_names if col not in {"prompt", "generation"}])
        data = data.map(lambda x: { # type: ignore
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['prompt'] if len(x['prompt'])<1024 else x['prompt'][:1024]}
            ],
            'answer': extract_hash_answer(x['generation'])
        }) # type: ignore
        # Print one example
        print(data[0]['prompt'])
        print(data[0]['answer'])
        return data

    def generate_r1_prompt(instruction, target, tokenizer):
        r1_prefix = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': instruction}
        ]
        return {
            "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
            "answer": target
        }

    def get_coding(split="train", tokenizer=None) -> Dataset:
        data = load_dataset('open-r1/codeforces-cots', 'solutions')[split]  # type: ignore

        # Keep only 'prompt' and 'generation' fields
        data = data.remove_columns([col for col in data.column_names if col not in {"prompt", "generation"}])

        # Apply the transformation
        dataset = data.map(
            lambda x: generate_r1_prompt(x["prompt"], extract_hash_answer(x["generation"]), tokenizer),
            remove_columns=data.column_names
        )

        # Split the dataset
        train_test_split = dataset.train_test_split(test_size=0.1)

        # Print a sample
        print(train_test_split['train'][0])
        return train_test_split['train'], train_test_split['test']
    
    return get_coding_nt(partition)