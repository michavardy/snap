from pathlib import Path
import sys
import json
from typing import List, Type, TypeVar, Any, Dict, get_origin, Optional
import re
from langchain_core.prompts import PromptTemplate
sys.path.append(str(Path.cwd()))
from pydantic import BaseModel, Field

def clean_response(string:str) -> str:
    replace_dict = {
        "```":"",
        "```json": "",
        "json":"",
        "\\n": "",
        "\n": "",
        "\'": "",
        }
    for key, value in replace_dict.items():
        string = string.replace(key, value)
    return string.strip()
def clean_regex_input(string:str) -> str:
    replace_dict = {
        "{":"",
        "}":"",
        "\"":"",
        "\n":"",
        }
    for key, value in replace_dict.items():
        string = string.replace(key, value)
    string = string.strip()
    return string
def clean_regex_output(string:str) -> str:
    replace_dict = {
        "\"":"",
        "\{":"",
        "\}":"",
        "]}":"]",
        "\n":"",
        "```":"",
        ".,":"."
        }
    for key, value in replace_dict.items():
        string = string.replace(key, value)
    string = string.strip()
    return string
def clean_regex_value(string:str) -> str:
    replace_dict = {
        "true,":"true"
        }
    for key, value in replace_dict.items():
        string = string.replace(key, value)
    string = string.strip()
    return string
def parse_response_regex(response_str:str, response_model: Type[BaseModel]) -> BaseModel:
    regex_pattern = re.compile("".join([f"{required_variable}.*?:(?P<{required_variable}>.*)" for required_variable in response_model.model_json_schema().get("properties",{}).keys()]), re.MULTILINE | re.DOTALL)
    cleaned_response = clean_response(response_str)
    cleaned_response = clean_regex_input(cleaned_response)
    match = re.search(regex_pattern,cleaned_response)
    json_response = {key:value.strip() for key,value in match.groupdict().items()}
    partial_response_json = {}
    for key,value in json_response.items():
        value = clean_regex_value(value)
        property_type = response_model.model_fields[key].annotation
        if type(value) == property_type:
            partial_response_json[key] = value
            continue
        try:
            typed_value = json.loads(value)
            assert (type(typed_value) == property_type) or (type(typed_value) == get_origin(property_type))
        except:
            continue
        partial_response_json[key] = typed_value
    return response_model(**partial_response_json)    
def parse_response(response_str:str, response_model: Type[BaseModel]) -> BaseModel:
    cleaned_response = clean_response(response_str)
    keys = response_model.model_json_schema()['properties'].keys()
    if len(keys) == 1 and response_model.model_fields[[k for k in keys][0]].annotation == str:
        key = [k for k in keys][0]
        return response_model(key=cleaned_response)
    json_response = json.loads(cleaned_response)
    return response_model(**json_response)
def parse_response_tokens(response_str:str, response_model: Type[BaseModel]) -> BaseModel:
    response_str = response_str.removeprefix("```json").removesuffix("```").strip()
    response_str = response_str[response_str.find("{") + 1: response_str.rfind("}")]
    fields = response_model.model_json_schema().get("properties", {}).keys()
    indexes = [(f, response_str.find(f"\"{f}\":")) for f in fields]
    indexes.sort(key=lambda x: x[1])

    res = {}
    for i, e in enumerate(indexes):
        key = e[0]
        if i < len(indexes) - 1:
            value = response_str[e[1] + len(f"\"{key}\":"): indexes[i + 1][1]]
        else:
            value = response_str[e[1] + len(f"\"{key}\":"):]
        res[key] = value.strip().removeprefix("\"").removesuffix("\"").strip()

    for k in response_model.model_fields.keys():
        t = response_model.model_fields[k].annotation
        if t == str:
            res[k] = json.loads(json.dumps(res[k]))
        elif t == list[str]:
            res[k] = json.loads(res[k].replace("'", "\""))
        else:
            res[k] = json.loads(res[k])

    return response_model(**res)

def get_response(llm: Any, prompt: PromptTemplate, response_model: Type[BaseModel],params:Dict[str,Any]={}, injection:bool = True) -> BaseModel:
    parsing_functions = [
        parse_response,
        parse_response_tokens,
        parse_response_regex,
        ]
    response_model_schema = json.dumps(response_model.model_json_schema())
    prompt_injection = """
    Output Schema:
    -------------
    Please provide the output in the following structured format:
    don't output the schema, just the response in the following format:
    ```schema
    response_model_schema
    ```
    """
    prompt_injection = prompt_injection.replace("response_model_schema",response_model_schema).replace("{","{{").replace("}","}}")
    if injection:
        prompt.template = f"{prompt.template}\n{prompt_injection}"
        prompt.input_variables = [key for key in params.keys()]
    chain = prompt | llm
    response = chain.invoke(params).content
    
    for response_function in parsing_functions:
        try:
            return response_function(response,response_model)
        except Exception:
            continue
    raise Exception("Failed to parse structured response")

if __name__ == "__main__":
    pass
    
    