from typing import Any, Dict
from dataclasses import dataclass, is_dataclass

def get_type_structure_recurse(obj: Any) -> Dict:
    if is_dataclass(obj):
        obj = vars(obj)  # If it's a dataclass, get its fields
    
    obj_types = {}
    
    for name, value in obj.items():
        if isinstance(value, (dict, list)):  # Check if the value is a dictionary or list
            nested_obj_types = get_type_structure_recurse(value)  # Recurse into it
            obj_types[name] = nested_obj_types
        else:
            obj_types[name] = type(value)  # Store the type of the value
            
    return obj_types

@dataclass
class Test:
    a: int
    b: str
    c: dict
            
obj1 = Test(1, "test", {"d": 2, "e": "test2"})
obj_types = get_type_structure_recurse(obj1)
print(obj_types)
