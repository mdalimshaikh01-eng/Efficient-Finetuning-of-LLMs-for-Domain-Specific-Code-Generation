import os
import requests
from datetime import datetime


# Function to ensure directory exists within the 'results' folder
def ensure_results_directory_exists(directory):
    results_directory = os.path.join('results', directory)
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)


url = "http://localhost:11434/api/generate"
model = "my_llama3.1:8b-instruct-q5_K_M"
file_name = 'rating.txt'
model_file= 'modelfile'

# Load the tokenizer and model
#model = "meta-llama/Meta-Llama-3.1-8B-Instruct"


try:
    # Read the prompt from a text file
    with open(file_name, 'r') as file:
        prompt_text = file.read()

except FileNotFoundError:
    print("Error: The prompt file 'prompt.txt' was not found.")
    #prompt_text = ""  # Set prompt_text to empty if file not 
    #  to prevent further errors

try:
    # Read the prompt from a text file
    with open(model_file, 'r') as file:
        model_text = file.read()
        
except FileNotFoundError:
    print("Error: The prompt file 'prompt.txt' was not found.")

data = {
    "model": model,
    "prompt": prompt_text,
    "stream": False
}

headers = {
    "Content-Type": "application/json"  # Specify the content type as JSON
}

def save_output_to_file(response_text):
    timestamp = datetime.now().strftime("%Y/%m/%d_%H:%M:%S")
    date_directory = timestamp.split('_')[0].replace('/', os.sep)
    result_dir = 'result'
    if not os.path.exists(os.path.join(result_dir, date_directory)):
        os.makedirs(os.path.join(result_dir, date_directory))
    filename = os.path.join(result_dir,date_directory, f'ratings_{timestamp.split("_")[1]}.txt')
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(model)
        file.write('\n\n')
        file.write(file_name)
        file.write('\n\n')
        file.write(model_text)
        file.write('\n\n')
        file.write(prompt_text)
        file.write('\n\n')
        file.write('=======================================================================================================================================================================')
        file.write('\n\n')
        file.write(response_text)
    print(f"Response saved to '{filename}'")

if __name__ == "__main__":
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors (status codes 4xx or 5xx)
        
        response_json = response.json()
        response_text = response_json["response"]
        # Call the function to save output to file
        save_output_to_file(response_text)
    except requests.exceptions.RequestException as e:
        print("Error occurred:", e)
