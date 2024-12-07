from pathlib import Path
import sys
from typing import Any, List, Type, Dict, Tuple, TypedDict, Optional, get_type_hints
import json
from pydantic import BaseModel, create_model
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
sys.path.append(str(Path.cwd()))
from utils.multi_thread_utils  import get_multi_threaded
from utils.parser_utils import get_response
load_dotenv()
llm = ChatOpenAI()
from tqdm import tqdm


class Param(TypedDict):
    data_point: str
    params: dict
    
class Label(TypedDict):
    data_point: str
    label: BaseModel

# Function to create a dynamic Pydantic model from JSON data sample
def create_dynamic_model_from_data(name: str, sample_data: Dict[str, Any]) -> Type[BaseModel]:
    fields = {key: (type(value), ...) for key, value in sample_data.items()}
    return create_model(name, **fields)
def get_example_dataset(dataset_json: dict) -> List[BaseModel]:
    dataset_objects=[]
    for example_data_point in tqdm(dataset_json['dataset'], desc=f"Creating Example Data Points"):
        DataModel, LabelModel, DataPoint, ParamModel = get_models(example_data_point)
        example_dataset = get_datapoint_instance(example_datapoint=example_data_point, DataModel=DataModel, LabelModel=LabelModel, DataPoint=DataPoint, ParamModel=ParamModel)
        dataset_objects.append(example_dataset)
    return dataset_objects
def get_models(example_datapoint: dict) -> Tuple[Type[BaseModel], Type[BaseModel], Type[BaseModel]]:
    DataModel = create_dynamic_model_from_data("Data", {
        "data": example_datapoint["data"],
        "data_field_name": example_datapoint["data_field_name"],
        "data_variation_name": example_datapoint["data_variation_name"],
    })
    LabelModel = create_dynamic_model_from_data("Label", example_datapoint["label"])
    ParamModel = create_dynamic_model_from_data("Param", example_datapoint["params"]) if "params" in example_datapoint else None
    if ParamModel:
        DataPoint = create_model("DataPoint", data=(DataModel, ...), label=(LabelModel, ...), params=(ParamModel, ...))
    else:
        DataPoint = create_model("DataPoint", data=(DataModel, ...), label=(LabelModel, ...))
    return DataModel, LabelModel, DataPoint, ParamModel
def get_datapoint_instance(example_datapoint: dict, DataModel: Type[BaseModel], LabelModel: Type[BaseModel], DataPoint: Type[BaseModel], ParamModel: Optional[Type[BaseModel]] = None) -> List[BaseModel]:
    data_instance = DataModel(**example_datapoint)  # Unpack item into DataModel
    label_instance = LabelModel(**example_datapoint['label']) # Unpack first label in the list into LabelModel
    param_instance = ParamModel(**example_datapoint['params']) if 'params' in example_datapoint and ParamModel else None
    if param_instance:
        data_point_instance = DataPoint(data=data_instance, label=label_instance, params=param_instance)
    else:
        data_point_instance = DataPoint(data=data_instance, label=label_instance)

    return data_point_instance
def format_element(element: str) -> str:
    return element.strip().replace("\n","")
def get_type_structure_recurse(obj: Any) -> Dict:
    if isinstance(obj, BaseModel):
        obj = vars(obj)  
    obj_types = {}
    for name, value in obj.items():
        if isinstance(value, (dict, list)):  # Check if the value is a dictionary or list
            nested_obj_types = get_type_structure_recurse(value)  # Recurse into it
            obj_types[name] = {key:type(value).__name__ for key,value in nested_obj_types.items()}
        else:
            obj_types[name] = type(value).__name__  # Store the type of the value   
    return obj_types  
def get_example_data_summary(examples: List[BaseModel]) -> str:
    examples_data_text =  "\n---\n".join([f"\ndata:\n----\n{data.data.data}".replace("\\\\","\\") for data in examples])
    field_name = "\n---\n".join([f"\nfield_name:\n----\n{data.data.data_field_name}".replace("\\\\","\\") for data in examples])
    prompt = PromptTemplate(
        template= """
        Prompt:
        Given the example data, provide a concise 1-2 sentence explanation. 
        The first sentence should describe the general function or purpose of the data in broad terms 
        (e.g., “provides information about an event”), 
        while the second sentence should specify what insights or details the data is conveying. 
        Avoid mentioning format or structure.
        do not use the word data or discuss input, explination or text in your explanation.
        be very brief, clear and terse

        Input:
        Examples: {examples_data_text}
        field_name: {field_name}

        Output:
        Explanation: str
        """,
        input_variables=["examples_data_text", "field_name"],
    )
    chain = prompt | llm
    response = chain.invoke({"examples_data_text":examples_data_text, "field_name":field_name})
    return response.content
def get_params_explination(examples: List[BaseModel]) -> str:
    examples_text =  "\n---\n".join([f"\ndata:\n----\n{data.data.data},\n\nparams:{data.params}".replace("\\\\","\\") for data in examples])
    data_type = type(examples[0].data.data)
    params_type = "\n".join([f"{field_name} :({field_info.annotation.__name__})" for field_name, field_info in examples[0].params.__fields__.items()])
    prompt = PromptTemplate(
        template= """
        given the examples_text, data_type and params_type, explain in 1-2 sentences what is the connection between the data and the params.
        The first sentence should describe what the params represents and the  purpose of the params
        while the second sentence should specify what exactly is the relationship between the data, and params
        Avoid mentioning format or structure.
        do not use the word data or discuss input, explination or text in your explanation.
        be very brief, clear and terse
        Input:
        -----
        Examples: {examples_text}
        
        Output Types:
        ------------
        Data: {data_type}
        params: {params_type}
    
        """,
        input_variables=["examples_text", "data_type", "params_type"],
    )
    chain = prompt | llm
    response = chain.invoke({"examples_text":examples_text, "data_type":data_type, "params_type":params_type})
    return response.content
def get_label_explination(examples: List[BaseModel]) -> str:
    if hasattr(examples[0], "params"):
        examples_text =  "\n---\n".join([f"\ndata:\n----\n{data.data.data},\n\nparams:{data.params}\n\nlabel:\n-----\n{json.dumps(data.label.__dict__)}".replace("\\\\","\\") for data in examples])
    else:
        examples_text =  "\n---\n".join([f"\ndata:\n----\n{data.data.data},\n\nlabel:\n-----\n{json.dumps(data.label.__dict__)}".replace("\\\\","\\") for data in examples])
    data_type = type(examples[0].data.data)
    label_type = "\n".join([f"{field_name} :({field_info.annotation.__name__})" for field_name, field_info in examples[0].label.__fields__.items()])
    prompt = PromptTemplate(
        template= """
        given the examples_text, data_type and label_type, explain in 1-2 sentences what is the connection between the data and the label.
        The first sentence should describe what the label represents and the  purpose of the label
        while the second sentence should specify what exactly is the relationship between the data, params and label
        Avoid mentioning format or structure.
        do not use the word data or discuss input, explination or text in your explanation.
        be very brief, clear and terse
        Input:
        -----
        Examples: {examples_text}
        
        Output Types:
        ------------
        Data: {data_type}
        Label: {label_type}
    
        """,
        input_variables=["examples_text", "data_type", "label_type"],
    )
    chain = prompt | llm
    response = chain.invoke({"examples_text":examples_text, "data_type":data_type, "label_type":label_type})
    return response.content
def get_dataset(data_summary: str, examples: List[BaseModel], n:int = 10) -> List[str]:
    examples_text =  "\n---\n".join([f"\ndata:\n----\n{data.data.data}".replace("\\\\","\\") for data in examples])
    data_type = type(examples[0].data.data)
    rounds = int(n*0.1)
    dataset = []
    prompt = PromptTemplate(
        template= """
        given the example data, data_type and data summary
        generate {k} datapoints that align with the data summary, faithful to the data type
        use the examples to guide the data generation
        
        guidelines:
        ----------
        - don't number the datapoints
        - do not describe the data or its structure
        - do not explain the summary
        - output must be a valid instance of the data 
        - use | symbol to deliniate between datapoints
        - do not format the sentences
        - do not use \n
        
        Input:
        -----
        Examples: {examples_text}
        data_summary: {data_summary}
        
        Output Types:
        ------------
        data_type: {data_type}
    
        """,
        input_variables=["examples_text", "data_summary", "data_type", "k"],
    )
    for i in tqdm(range(int(n/rounds)), desc="generating dataset"):
        chain = prompt | llm
        response = chain.invoke({"examples_text":examples_text, "data_summary":data_summary, "data_type":data_type, "k":rounds})
        dataset.extend([format_element(element) for element in response.content.split("|")])
    return dataset
def get_params(data_set: list[str], examples: List[BaseModel], data_summary: str, params_explination:str) -> dict:
    examples_text =  "\n---\n".join([f"\ndata:\n----\n{data.data.data}\n\nparams:\n------\n{data.params}".replace("\\\\","\\") for data in examples])
    params_type = get_type_structure_recurse(obj=examples[0].params)
    prompt = PromptTemplate(
    template= """
    Task:
    ----
    Generate Params

    Instructions:
    ------------
    Given the datapoint, examples text, data_summary, and params explanation,
    generate parameters matching the specified `params_type`. 

    Guidelines:
    -----------
    - Do not add any formatting other than the parameter types.
    - Do not use `\n` in the output.
    - Do not output the data point or types.
    - Ensure the output strictly matches the parameter fields as per `params_type`.
    - output values not field types
    - guess the appropriate values

    Input:
    -----
    datapoint: {datapoint}
    Examples: {examples_text}
    data_summary: {data_summary}
    params_explination: {params_explination}

    Output Types:
    ------------
    params_type: {params_type}

    Expected Output:
    ----------------
    {params_type} should be used to generate the correct parameter values.
    """,
        input_variables=["datapoint", "examples_text", "data_summary", "params_explination", "params_type"],
    )
    params_dict_list = []
    for datapoint in tqdm(data_set, desc="generating params"):
        chain = prompt | llm
        response = chain.invoke({"datapoint":datapoint, "examples_text":examples_text, "data_summary":data_summary, "params_explination":params_explination, "params_type":params_type})
        params = Param(**json.loads(response.content.replace("\'","\"")))
        params_dict_list.append({"datapoint":datapoint, "params":params})
    return params_dict_list
def get_labels(data_set: List[str], examples: List[BaseModel], data_summary: str, label_map_summary: str) -> List[str]:
    examples_text =  "\n---\n".join([f"\ndata:\n----\n{data.data.data}\n\nLabels:\n------\n{data.label}".replace("\\\\","\\") for data in examples])
    label_type = get_type_structure_recurse(obj=examples[0].label)
    prompt = PromptTemplate(
    template= """
    Task:
    ----
    Generate Params

    Instructions:
    ------------
    Given the datapoint, examples text, data_summary, and label explination,
    generate labels matching the specified `label_type`. 
    the label should not be identical to example labels but should be similar in structure and content

    Guidelines:
    -----------
    - Do not add any formatting other than the parameter types.
    - Do not use `\n` in the output.
    - Do not output the data point or types.
    - Ensure the output strictly matches the parameter fields as per `params_type`.
    - output values not field types
    - guess the appropriate values

    Input:
    -----
    datapoint: {datapoint}
    Examples: {examples_text}
    data_summary: {data_summary}
    label_explination: {label_map_summary}

    Output Types:
    ------------
    label type: {label_type}

    Expected Output:
    ----------------
    {label_type} should be used to generate the correct parameter values.
    """,
        input_variables=["datapoint", "examples_text", "data_summary", "label_map_summary", "label_type"],
    )
    label_dict_list = []
    for datapoint in tqdm(data_set, desc="generating labels"):
        chain = prompt | llm
        response = chain.invoke({"datapoint":datapoint, "examples_text":examples_text, "data_summary":data_summary, "label_map_summary":label_map_summary, "label_type":label_type})
        try:
            label = Label(**json.loads(response.content.replace("\'","\"").replace("\n","").strip()))
        except:
            breakpoint()
        label_dict_list.append({"datapoint":datapoint, "label":label})
    return label_dict_list
def get_examples(data_point:BaseModel, example_dataset:list[BaseModel]) -> List[BaseModel]:
    examples = []
    for example in example_dataset:
        if example.data.data_variation_name == data_point.data.data_variation_name:
            examples.append(example)
    return examples
def generate_datapoint(data_point:BaseModel, n:int = 20, example_dataset:list[BaseModel]=[]) -> BaseModel:
    if hasattr(data_point, "params"):
        examples = get_examples(data_point, example_dataset)
        data_summary = get_example_data_summary(examples)
        params_map_summary = get_params_explination(examples)
        label_map_summary = get_label_explination(examples)
        data_set = get_dataset(data_summary, examples, n)
        params = get_params(data_set, examples, data_summary, params_map_summary)
        labels = get_labels(data_set, examples, data_summary, label_map_summary)
        data_dicts = [value if type(value)==dict else {key:value.__dict__} for key, value in examples[0].__dict__.items()]
        DataModel, LabelModel, DataPoint, ParamModel = get_models({**data_dicts[0]['data'], **data_dicts[1], **data_dicts[2]})
        data = [DataModel(data=data, data_field_name=examples[0].data.data_field_name, data_variation_name=examples[0].data.data_variation_name) for data in data_set]
        params = [ParamModel(**param['params']) for param in params]
        labels = [LabelModel(**label['label']) for label in labels]
        datapoints = [DataPoint(data=data, label=label, params=param) for data, label, param in zip(data, labels, params)]
    else:
        examples = get_examples(data_point, example_dataset)
        data_summary = get_example_data_summary(examples)
        label_map_summary = get_label_explination(examples)
        data_set = get_dataset(data_summary, examples, n)
        labels = get_labels(data_set, examples, data_summary, label_map_summary)
        data_dicts = [value if type(value)==dict else {key:value.__dict__} for key, value in examples[0].__dict__.items()]
        breakpoint()
        DataModel, LabelModel, DataPoint, ParamModel = get_models({**data_dicts[0]['data'], **data_dicts[1]})
        data = [DataModel(data=data, data_field_name=examples[0].data.data_field_name, data_variation_name=examples[0].data.data_variation_name) for data in data_set]
        labels = [LabelModel(**label['label']) for label in labels]
        datapoints = [DataPoint(data=data, label=label) for data, label in zip(data, labels)]
    return datapoints
def filter_unique_variation_names(example_dataset: list[BaseModel]) -> List[BaseModel]:
    variation_name_set =set()
    iterable_dataset = []
    for datapoint in example_dataset:
        if datapoint.data.data_variation_name not in variation_name_set:
            variation_name_set.add(datapoint.data.data_variation_name)
            iterable_dataset.append(datapoint)
    return iterable_dataset
def get_results_json(results: List[BaseModel]) -> List[Dict]:
    results_json = []
    for result in results:
        result_dict = {}
        for key in result.__dict__.keys():
            result_dict[key] = result.__dict__[key].__dict__
        results_json.append(result_dict)
    return results_json
if __name__ == "__main__":
    # Load JSON dataset``
    dataset_json = json.loads(Path("examples/people_weight.json").read_text())
    example_dataset = get_example_dataset(dataset_json)
    iterable_dataset = filter_unique_variation_names(example_dataset)
    dataset = get_multi_threaded(
            callback_function=generate_datapoint,
            iterable=iterable_dataset,
            max_workers=8,
            n = 20,
            example_dataset = example_dataset
        )
    results = get_results_json(dataset[0].get('result'))
    Path('examples/people_weight_dataset.json').write_text(json.dumps(results, indent=4))

    