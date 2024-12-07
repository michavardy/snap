import sys
from pathlib import Path
from langchain_core.prompts import PromptTemplate
sys.path.append(str(Path.cwd()))
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Type, Optional, Tuple, TypedDict, Any
from pydantic import BaseModel, ValidationError, Field
from utils.setup_logger import setup_logger
from utils.parser_utils import get_response
from tqdm import tqdm
from functools import partial
import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


logger = setup_logger(name=__name__)


class Result(TypedDict):
    item: Any
    result: Any

def get_multi_threaded(
    callback_function: Callable,
    iterable: Iterable,
    max_workers: int = 8,
    *args,
    **kwargs
) -> List[Result]:
    """
    Execute a callback function in a multi-threaded manner and return results along with their input items.

    Args:
        callback_function (Callable): The function to execute for each item in the iterable.
        iterable (Iterable): A collection of items to be processed.
        max_workers (int, optional): The maximum number of threads to run concurrently. Defaults to 8.
        *args: Additional positional arguments to pass to the callback function.
        **kwargs: Additional keyword arguments to pass to the callback function.

    Returns:
        List[dict]: A list of dictionaries, where each dictionary contains:
            - "item": The input item from the iterable.
            - "result": The corresponding output of the callback function.
    """
    results = []
    errors = []

    # Prepare the callback function with additional arguments
    wrapped_function = partial(callback_function, *args, **kwargs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(wrapped_function, item): item for item in iterable}

        for future in tqdm(as_completed(futures), desc="Processing instances", total=len(futures)):
            item = futures[future]
            try:
                result = future.result()
                result_dict = {"item":item, "result":result}
                results.append(result_dict)
                
            except ValidationError as ve:
                logger.error(f"Validation error for item {item}: {ve}")
                errors.append((item, ve))
                
            except Exception as e:
                logger.error(f"Error processing item {item}: {e}")
                errors.append((item, e))

    logger.info(f"Processing complete. Successes: {len(results)}, Failures: {len(errors)}")
    return results
def multi_threaded_apply( callback_function: Callable, df: pd.DataFrame,  max_workers: int = 8, *args, **kwargs) -> pd.Series:
    """
    Apply a function to each row of a DataFrame using multi-threading and return a DataFrame with the results.

    Args:
        callback_function (Callable): The function to apply to each row of the DataFrame. 
            The function should accept a row as input and return a result.
        df (pd.DataFrame): The input DataFrame whose rows will be processed.
        max_workers (int, optional): The maximum number of threads to run concurrently. Defaults to 8.
        *args: Additional positional arguments to pass to the callback function.
        **kwargs: Additional keyword arguments to pass to the callback function.

    Returns:
        pd.Series: A new Series containing a single column  with the results of applying 
            the callback function to each row of the input DataFrame.

    Notes:
        - The function relies on `get_multi_threaded` to handle multi-threaded processing.
        - The `callback_function` should be compatible with a row-like input (e.g., a Pandas Series).
    """
    rows = [row for index, row in df.iterrows()]
    results = get_multi_threaded(callback_function=callback_function, iterable=rows, max_workers=max_workers, *args, **kwargs)
    data = []
    for result in results:
        result.get('item')['new'] = result.get('result')
        data.append(result.get('item'))
    df = pd.DataFrame(data)
    return df['new']
def setup_test() -> Tuple[Callable, List, pd.DataFrame]:
    data = {
        "email_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "email_content": [
            "Thanks for reaching out! We'd love to set up a demo with your team. When are you available?",
            "Your prices are too high for our budget. Can we negotiate a discount?",
            "We’re thrilled to partner with you. Looking forward to the next steps!",
            "The last order was delayed, and our operations suffered. Can you ensure it won’t happen again?",
            "Thank you for the quick resolution to our issue. Much appreciated!",
            "Our team is still considering your proposal. Can you extend the deadline?",
            "Excellent service! We’ll be placing another order soon.",
            "We are very dissatisfied with the support provided. It was unprofessional."
        ],
        "customer_annual_revenue": [50000, 200000, 100000, 75000, 120000, 80000, 95000, 30000],
        "customer_size": ["small", "large", "medium", "small", "medium", "small", "medium", "small"]
    }
    email_df = pd.DataFrame(data)
    rows = [row for index, row in email_df.iterrows()]
    response_model = pd.Series
    
    def get_sentiment(row: pd.Series, k:float, *args, **kwargs) -> float:
        class Sentiment(BaseModel):
            sentiment: float = Field(0.5, description="sentiment score is a float between 0-1, 0 being very negative and 1 being very positive")
            
        prompt = PromptTemplate(
            template="""
                return the sentiment of the email content
                
                input:
                email_content: {email_content}
                customer_size: {customer_size}
            """,
            input_variables=["email_content", "customer_size"],
        )
        response = get_response(
            llm=kwargs.get('llm'), 
            prompt=prompt, 
            response_model=Sentiment, 
            params={"email_content": row["email_content"], "customer_size": row["customer_size"]}
            )
        return response.sentiment * k
    return get_sentiment, rows, email_df

if __name__ == "__main__":
    get_sentiment, rows, email_df = setup_test()
    llm = ChatOpenAI()
    email_df["sentiment_score"] = multi_threaded_apply(callback_function=get_sentiment, df=email_df, k=1.1, llm=llm)
    #results = get_multi_threaded(callback_function=get_sentiment, iterable=rows, max_workers=8, k=1)
    #
    #data = []
    #for result in results:
    #    result.get('item')['sentiment'] = result.get('result')
    #    data.append(result.get('item'))
    #df = pd.DataFrame(data)
    breakpoint()
    