##Load dataset
import os
import re
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from huggingface_hub import login
import csv
#from huggingface_hub import notebook_login


directory_path = 'LXOXF data'
data = []
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            data.append(content)


##Split data into steps and python code
steps_list = []
code_list = []

# Define the pattern to match steps and code separated by a line of dashes
pattern = re.compile(r'(?P<steps>.*?)(?P<code>.*)', re.DOTALL)

for content in data:
    parts = content.split('----------------------------------------------------------------------------------------------------------------')
    
    if len(parts) == 2:
        steps, code = parts
        steps_list.append(steps.strip())
        code_list.append(code.strip())
##Clean and normalize data
def clean_text(text):
    text = text.lower()  
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text.strip()

steps_list = [clean_text(step) for step in steps_list]
code_list = [clean_text(code) for code in code_list]

model = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.getenv(model),
    add_eos_token=True, 
    trust_remote_code=True
)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.truncation_side="left"
# Tokenize the data
def tokenize_data(text_list, tokenizer):
    tokenizers = tokenizer(text_list)
    tokenizers.pad_token=tokenizer.eos_token
    tokenizers.truncation_side="left"
    return 

# Tokenize steps and code
steps_encodings = tokenize_data(steps_list, tokenizer)
code_encodings = tokenize_data(code_list, tokenizer)


#print(code_encodings)

'''with open("output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # Write rows to CSV file
    # Each column is formed by taking one list from each main list
    for i in range(len(list1)):
        row = list1[i] + list2[i]  # Concatenate corresponding sublists
        writer.writerow(row)'''