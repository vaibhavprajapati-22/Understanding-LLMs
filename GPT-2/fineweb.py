import os
import numpy as np
import tiktoken
from datasets import load_dataset

# Load the GPT-2 tokenizer
enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens['<|endoftext|>']  # end of text token

# Load the dataset
remote_name = "sample-10BT"
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)

target_token_count = 100_000_000  # First 100M tokens
current_token_count = 0  # Track total token count
token_buffer = [eot]  # Buffer to store tokens

# Directory to save the tokens
local_directory = "fineweb_edu_tokenized"
if not os.path.exists(local_directory):
    os.makedirs(local_directory)


# Function to save the tokens to a single NumPy array file
def save_tokens_as_np(tokens):
    filename = os.path.join(local_directory, f"tokenized_100M.npy")
    np_array = np.array(tokens, dtype=np.uint16)
    np.save(filename, np_array)
    print(f"Saved {len(tokens)} tokens to {filename}")


# Tokenize the dataset and collect tokens
for example in fw:
    # Tokenize the text from the example (assuming the dataset has a 'text' field)
    tokens = enc.encode(example['text'])

    # Add tokens to buffer
    token_buffer.extend(tokens)
    current_token_count += len(tokens)

    # Stop when we hit 100M tokens
    if current_token_count >= target_token_count:
        # Trim excess tokens if we've exceeded 100M tokens
        token_buffer = token_buffer[:target_token_count]
        break

# Save all tokens as a single NumPy array
save_tokens_as_np(token_buffer)

print(f"Finished tokenizing and saving {len(token_buffer)} tokens.")
