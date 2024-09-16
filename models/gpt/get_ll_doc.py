import json
import re
from openai import OpenAI
from tqdm import tqdm

PATH = "M2L_PATH"

KEY = {
    "api_key": "API_KEY",
    "api_base": "API_BASE"
}
CLIENT = OpenAI(api_key=KEY['api_key'], base_url=KEY['api_base'])

GET_DOC = '''\n\nGiven the dictionary above that describes a function that performs a low-level action, provide formal 
             documentation for the purpose of the function and what each argument is for. The argument in the above 
             dictionary is a sample argument, you may use it to reason what each argument is meant for in the function;
             however, you must provide your documentation devoid of referring to the specific argument provided in the 
             example above. Provide the documentation in string. you may output the function as follows 
             function_name(<arg1>, <arg2>, ...) etc and use this to refer to part of the function when making 
             descriptions. Output your response in a single concise paragraph. Do not provide any line breaks.'''

def gpt(prompt):
    try:
        output = CLIENT.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt}
            ]
        )
        return output.choices[0].message.content
    except:
        return "error occurred"

def get_function_arg(function):
    pattern = r'(\w+)\((.*)\)'
    match = re.match(pattern, function)
    
    if match:
        return {"function_name" : match.group(1).lower().strip(),
                "argument" : [arg.lower().strip() for arg in match.group(2).split(",")]}
    else:
        return {"function_name" :  None, "argument" : None}

def main():
    doc_path  = "FUNCTION_PATH"
    doc = []
    with open(doc_path, "r") as f:
        doc =  json.load(f)
    ids = list(doc.keys())

    for vid in tqdm(ids[:2]):
        documentation  = ""
        for func in tqdm(doc[vid]['parsed_function'], leave=False):
            documentation += gpt(GET_DOC + str(func)) + "\n"
        doc[vid]["documentation"] = documentation

    with open("SAVE_PATH", "w")  as f:
        json.dump(doc, f, indent=4)
        
if __name__ == "__main__":
    main()

