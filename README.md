# Snap

## Description
this is a CLI application that takes a set of example data and generates a dataset of similar data.

## Explination
this application is designed to take a set of labled example data and generate a labled dataset of similar data with the same structure. 

the example input data is a json object that contains the following fields:
- data: the data field
- data_field_name: the name of the data field
- data_type: if there are multiple types of data that can be generated from the same data field, this field is used to specify the name of this specific one
- params: the parameters that are used to generate the result (label)
- label: the result field

## Usage
1. Simple Example Data Struture
```json
{
            "data": "good morning",
            "data_field_name": "greetings",
            "data_variation_name":null
            "data_params": {},
            "label_field_name": "response",
            "label": {
                "message": "thanks, you too"
            }
}
```

2. More Complex Example Data Structure
```json
{
            "data": "Thursday off but dinner at galias with nurses",
            "data_field_name": "message",
            "data_type_name": "next_week",
            "params": {
                "current_date": {
                    "year": 2024,
                    "week": 46,
                    "weekday": 1
                },
                "target_day": 4
            },
            "label": {
                "message": "thursday off but dinner at galias with nurses",
                "year": 2024,
                "week": 46,
                "weekday": 4
            }
}

```
