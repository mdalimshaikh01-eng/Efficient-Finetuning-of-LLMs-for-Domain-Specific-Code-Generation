import os
import warnings
import pandas as pd
from crewai import Agent, Task, LLM, Crew
from utils import load_vectordb, get_test_case_code
from langtrace_python_sdk import langtrace
from langtrace_python_sdk.utils.with_root_span import with_langtrace_root_span
faiss_path = "/home/ml/./ollamawork/RAG/faiss-Thales-updated"
search_folder='/home/ml/./ollamawork/RAG/New Framework Code'
df= pd.read_excel('/home/ml/./ollamawork/Finetuning_2025/Final_test_data_bhusan.xlsx')

vectordb = load_vectordb(faiss_path)
import subprocess
import time


def retrieved_content(query, search_folder, k=3):
    # Retrieve the top 3 most relevant documents
    docs_with_scores = vectordb.similarity_search_with_score(query, k)
    

    # Initialize the counter for test case numbering
    counter = 1
    output = []

    for doc, _ in docs_with_scores:
        first_line = doc.page_content.split('\n')[0]
        if first_line.startswith("Name:"):
            test_case_name = first_line.split(":")[1].strip()
            output.append(f"**Test Case {counter}: Generate a Python script based on the provided testing steps to verify test mention in the summary.**\n")
            output.append(doc.page_content)

            # Fetch and write the code
            code = get_test_case_code(test_case_name, search_folder)
            output.append(f"\nPython Code {counter}:\n###--Start of Python Code--\n{code}\n###--End of Python Code--\n")
            
            # Increment the counter for the next test case
            counter += 1
    
    output.append(f'###Test Case to be Automated: Generate a Python script based on the provided testing steps to verify test mention in the summary.\n')
    output.append(query)
    return '\n'.join(output)

from langtrace_python_sdk import langtrace

langtrace.init(api_key = 'http://localhost:3000')

warnings.filterwarnings('ignore')

n=2
@with_langtrace_root_span()
def my_agents():
    for i in range(10):
        query = f'''
Name: {df['Test Case Name'][i+n]}
Summary:  {df['Summary'][i+n]}
Test steps:
{df['Steps with Expected Output'][i+n]}
        '''

        # Define LLM instances
        def get_llm(model_name):
            return LLM(model=model_name, base_url="http://localhost:11434/api/generate", parameters={"seed": 42})
        
        def get_llm1(model_name):
            return LLM(model=model_name, base_url="http://10.171.3.97:11434/api/generate", parameters={"seed": 42})
      
        
        llm1 = get_llm1("ollama/phi4:review")
        llm = get_llm1("ollama/llama3.1_64k:latest")

        code_generator = Agent(
                role="Automated Test Case Generator",
                goal="Generate a complete Python code to automate the following test case to be automated based on the pattern in the given Test Case examples.",
                backstory="""
                    You are an expert Python developer specializing in automated testing. 
                    You analyze test case patterns and generate well-structured Python scripts 
                    with a step-by-step approach. Your focus is on clarity, maintainability, 
                    and adherence to PEP 8 coding standards. You ensure that each function 
                    is explicitly written and properly structured, following the approach 
                    used in previous test case examples. 
                """,
                allow_delegation=False,
                verbose=True,
                llm=llm
            )

        # Define the task with structured test cases
        generated_code = Task(
                description=(
                    '**Write each and every function of your code completely with exactly as it is done in Test Case examples and do not pass any function.'
                    'Do not directly use the functions and methods provided in the context. Instead, understand how they are used in the test cases and write all the required functions and methods explicitly to complete the steps in the test case'
                    'Use Chain of Thought while generating code for test case to be automated.' 
                    'Adheres to PEP 8 coding guidelines and uses snake_case for variable, function, and method names.**'
                    "### Reference Test Cases:\n\n"
                    + retrieved_content(query, search_folder, k=3)
                ),
                expected_output=(
                    "**Generated Python Code:**\n"
                    "The output script should follow the same format as previous test cases while ensuring adherence to PEP 8 standards and a structured approach. "
                ),
                agent=code_generator,
                output_file='1st_Generated_code.txt'
            )

        

        code_evaluator = Agent(
                role="Test Code Evaluator",
                goal="Evaluate the generated_code script by comparing it against the provided test steps and human-written similar type example.",
                backstory="""
                    You are helpful LLM as a judge for programming code evaluation assistant. Your task is to analyze and score the generated_code scripts based on their correctness, completeness, and adherence to test steps and similar test case.
                    
                    You will:
                    - Compare the generated code with the given similar test case and test steps.
                    - Identify any missing, incorrect, or unnecessary parts.
                    - Provide a numerical score (1-10) based on accuracy and usability.
                    - Use Chain of Thought reasoning to explain each issue in detail.
                    - Avoid redundant comments and ensure clarity in explanations.
                """,
                allow_delegation=False,
                verbose=True,
                llm=llm1
            )

        evaluator = Task(
                description=(
                    " Evaluate the generated_code based on the similar test case given below and test steps of the generated_code"

                     'You are helpful LLM as a judge for programming code evaluation assistant. Your task is to: '         
                    '- Identify the patterns in the human-written test cases steps and their corresponding code.' 
                    '- Evaluate the LLM-generated code by comparing it to the provided test steps and the patterns observed '   
                    'in the human-written examples'

                    


                    "#### Reference Test Cases:**\n"
                    "The following test cases include test steps and their corresponding human-written Python implementations. "
                    "Use these as a reference to evaluate the generated_code code.\n\n"

                    + retrieved_content(query, search_folder, k=1)
                    + "\n\n" 
                ),
                
                expected_output=(
                    "## ** Evaluation Report**\n\n"
                    'The first line should give the output score in the same format given below.'
                    
                    "** Score:** X/10\n\n"
                    'Give a detailed description of you evaluation following closely the Evaluation Guidelines and Scoring Guide given below.'

                    '### Evaluation Guidelines:'
                    '- **Score** the generated code on a scale of 1 to 10.'  
                    '- **Highlight all issues** in the generated code, including missing elements, incorrect logic, or deviations from the provided test steps.  '
                    '- **Provide a justification** for your score, focusing on alignment with the test steps and identifying any missing or incorrect parts.'
                    '- **Understand the comments between the code and log messages to map steps to the corresponding code blocks**'
                    '- **Avoid redundant comments; if the steps are clear, corresponding code block is present and the logic is correct, do not add comments.**'  
                    '- Use **Chain of Thought reasoning** to explain your analysis clearly.'

                    "### ** Scoring Guide (1-10)**\n"
                    "1-3 **Very Poor** - Major issues, hallucinations, or missing key steps. Needs significant modifications.\n"
                    "4-5 **Unusable** - Code is somewhat relevant but misses key objectives. Debugging takes longer than rewriting.\n"
                    "6-7 **Adequate** - Mostly usable but requires some modifications (e.g., incorrect assertions, incomplete logic).\n"
                    "8 **Good** - Achieves the test objective with only minor issues (e.g., logging errors, minor inconsistencies).\n"
                    "9-10 **Very Good** - Well-structured, accurate, and fully aligned with test steps. Can be executed without issues.\n\n"
                ),

                agent=code_evaluator,
                #context={'code_generator': code_generator, },
                output_file='code-evaluation1.txt'
            )
        crew1 = Crew(agents=[code_generator, code_evaluator],
                     tasks=[ generated_code, evaluator],
                       verbose=True,sequential=True)
        result1 = crew1.kickoff()
        #print(result1)
        import time
        '''print("Code is running...")
        time.sleep(60)  # Pause for 30 seconds
        print("Resumed after 30 seconds!")'''
        output_file = f"/home/ml/./ollamawork/Finetuning_2025/Crewai/RESULTS/output_26-3-25_1_{df['Test Case Name'][i+n]}_loop.txt"
        with open(output_file, "a") as outfile:
            outfile.write(query + "\n")
            for fname in ['1st_Generated_code.txt',"code-evaluation1.txt"]:
                with open(fname) as infile:
                    outfile.write(infile.read() + "\n")

        prev_score = 0
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            if iteration==0:
                with open('1st_Generated_code.txt', 'r') as f:
                    result1 = f.read()

                with open('code-evaluation1.txt', 'r') as f:
                    result2= f.read()

            else: 
                with open('code-generated'+str(iteration)+'_th.txt', 'r') as f:
                    result3 = f.read()

                with open('code-evaluation'+str(iteration)+'_th.txt', 'r') as f:
                    result4 = f.read()
            llm2 = get_llm("ollama/phi4_modelfile_gen:latest")
            Re_generated_code = Agent(
                role="Test Code Re-generator",
                goal="Re-generate complete corrected test code using the generated_code, the suggestions of the evaluator and one similar test case. While making changes, keep the overall structure and correct parts of the gewnerated_code the same. Only update sections with clear differences or those related to fixing the identified issues.",
                backstory="""
                    You are a skilled programmer tasked with improving the code. Use similar examples, the code generated by LLM, and the issues found in that code as a reference to create a better version.
                    The Automated Test Case Generator is designed to enhance and refine test cases based on provided test steps and LLM feedback. 
                    It operates as an intelligent assistant, ensuring that the test scripts remain accurate, efficient, and aligned with best practices.

                    It carefully analyzes discrepancies given by evaluator, preserves correctly functioning sections, and implements only necessary modifications to improve reliability.

                    By leveraging LLM evaluations and historical test cases, the agent ensures consistency while reducing redundancy.
                    This approach helps in maintaining a high level of test quality, reducing human intervention, and accelerating software validation processes.
                """,
                allow_delegation=False,
                verbose=True,
                llm=llm2
            )

            re_generated_code = Task(
                description=(
                    'Re-generated the generated_code by encorporating the suggestions given by evaluator. The test steps and one similartest code is given below for your suggestion. Update and improve the code using the similar test case, provided test steps, and the  evaluator output. While making changes, keep the overall structure and correct parts of the original code the same. Only update sections with clear differences or those related to fixing the identified issues.'
                    '### Siilar test case'
                    + retrieved_content(query, search_folder, k=1) +
                    '### Code and review'+
                    (str(result1 + result2) if iteration == 0 else str(result3 + result4))),
                
                expected_output=( 
                    "**Re-generated Python Code:**\n"
                    "Re-generated complete end to end test code. The re-generated test code should be fully accurate and optimized to achieve a higher score when reviewed again by the evaluator. "
                ),

                agent=Re_generated_code,
                output_file='code-generated'+str(iteration + 1)+'_th.txt'
            )
            '''print("Code is running...")
            time.sleep(60)  # Pause for 30 seconds
            print("Resumed after 30 seconds!")'''
 
            code_re_evaluator = Agent(
                role="Test Code Re-evaluator",
                goal="Re-evaluate the re-generated_code script by comparing it against the provided test steps and human-written similar type example.",
                backstory="""
                    You are helpful LLM as a judge for programming code evaluation assistant. Your task is to analyze and score the re-generated_code scripts based on their correctness, completeness, and adherence to test steps.
                    
                    You will:
                    - Compare the re-generated code with the given test steps.
                    - Identify any missing, incorrect, or unnecessary parts.
                    - Provide a numerical score (1-10) based on accuracy and usability.
                    - Use Chain of Thought reasoning to explain each issue in detail.
                    - Avoid redundant comments and ensure clarity in explanations.
                """,
                allow_delegation=False,
                verbose=True,
                llm=llm1
            )

            re_evaluator = Task(
                description=(
                    "Evaluate the re-generated_code based on the similar test case given below and test steps of the re-generated_code"

                    "#### Reference Test Cases:**\n"
                    "The following test cases include test steps and their corresponding human-written Python implementations. "
                    "Use these as a reference to evaluate the generated_code code.\n\n"

                    + retrieved_content(query, search_folder, k=1)
                    + "\n\n" + 
                    "### Generated code"

                    "## ** Evaluation Criteria & Guidelines**\n"
                    "**Your task is to analyze the generated_code code based on the following criteria:**\n\n"

                    " **1. Completeness:** Does the code implement all required test steps?\n"
                    " **2. Accuracy:** Are the function calls, logic, and expected results correct?\n"
                    " **3. Readability & Clarity:** Is the code structured properly with meaningful variable names and comments?\n"
                    " **4. Correct Logging & Assertions:** Are test logs and assertions present and correctly implemented?\n"
                    " **5. Error Handling:** Does the code manage errors properly, preventing failures due to unhandled exceptions?\n"
                    " **6. Consistency with Human-Written Code:** Does it follow a similar approach as previous test cases?\n\n"

                    "** Common Issues to Flag:**\n"
                    "- Missing or incorrectly implemented steps.\n"
                    "- Incorrect logic or unnecessary complexity.\n"
                    "- Inconsistent variable names, incorrect indentation, or improper function calls.\n"
                    "- Redundant or missing log messages and assertions.\n"
                    "- Unnecessary comments or lack of required documentation.\n\n"

                    "### ** Scoring Guide (1-10)**\n"
                    "1-3 **Very Poor** - Major issues, hallucinations, or missing key steps. Needs significant modifications.\n"
                    "4-5 **Unusable** - Code is somewhat relevant but misses key objectives. Debugging takes longer than rewriting.\n"
                    "6-7 **Adequate** - Mostly usable but requires some modifications (e.g., incorrect assertions, incomplete logic).\n"
                    "8 **Good** - Achieves the test objective with only minor issues (e.g., logging errors, minor inconsistencies).\n"
                    "9-10 **Very Good** - Well-structured, accurate, and fully aligned with test steps. Can be executed without issues.\n\n"
                ),
                
                expected_output=(
                    "## ** Evaluation Report**\n\n"
                    
                    'The first line should give the output score in the same format given below.'
                    "** Score:** X/10\n\n"

                    "** Identified Issues:**\n"
                    "Note down all the issues identified in the code."

                    "** Justification for Score:**\n"
                    "- **Completeness:** [Did the generated code cover all steps?]\n"
                    "- **Accuracy:** [Was the logic correct? Any incorrect assertions or function calls?]\n"
                    "- **Readability:** [Was the structure clear? Meaningful variable names? Proper indentation?]\n"
                    "- **Error Handling:** [Did it properly handle failures and unexpected inputs?]\n"
                    "- **Comparison with Human-Written Code:** [How does it compare to previous test cases?]\n\n"

                    "** Recommended Improvements:**\n"
                    "- **Fix missing elements:** [Specify missing functions or steps]\n"
                    "- **Correct incorrect logic:** [Point out incorrect calculations, loops, or assertions]\n"
                    "- **Enhance clarity:** [Suggest better variable names, comments, or structure improvements]\n\n"
                ),

                agent=code_evaluator,
                output_file='code-evaluation'+str(iteration + 1)+'_th.txt')

            crew2 = Crew(agents=[Re_generated_code, code_re_evaluator ], tasks=[re_generated_code, re_evaluator], verbose=True, sequential=True)
            result_2=crew2.kickoff()

            with open(output_file, "a") as outfile:
                outfile.write('\n\n\n\n')
                outfile.write('---------------------------------------------------------------------------------------------------------------\n\n')
                outfile.write('Iteration '+ str(iteration + 1))
                outfile.write('\n\n')
                for fname in ['code-generated'+str(iteration + 1)+'_th.txt','code-evaluation'+str(iteration + 1)+'_th.txt']:
                    with open(fname) as infile:
                        outfile.write(infile.read() + "\n")   
            import re

            with open('code-evaluation'+str(iteration + 1)+'_th.txt') as f:
                eval_content = f.read()
                new_score = int(re.search(r'(?i)\bscore\b.*?(\d+)', eval_content).group(1))

            if new_score <= prev_score:
                break
            prev_score = new_score
            iteration=iteration+1
my_agents()
