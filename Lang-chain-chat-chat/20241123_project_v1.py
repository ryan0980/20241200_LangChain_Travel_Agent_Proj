import os
from dotenv import load_dotenv
import pandas as pd
import json
import textwrap
from typing_extensions import List, TypedDict
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

print("OPENAI_API_KEY:", "*****" if os.getenv("OPENAI_API_KEY") else "Not Set")
print("LANGCHAIN_TRACING_V2:", os.getenv("LANGCHAIN_TRACING_V2"))
print("LANGCHAIN_ENDPOINT:", os.getenv("LANGCHAIN_ENDPOINT"))
print("LANGCHAIN_API_KEY:", "*****" if os.getenv("LANGCHAIN_API_KEY") else "Not Set")
print("LANGCHAIN_PROJECT:", os.getenv("LANGCHAIN_PROJECT"))
print("USER_AGENT:", os.getenv("USER_AGENT"))

# Initialize ChatOpenAI client
llm = ChatOpenAI(model="gpt-4", temperature=0)







def llm_intent_classification(question: str) -> bool:
    """
    Uses LLM to determine if the user's question is related to searching for hotels.
    Returns True if related to hotels, False otherwise.
    """
    prompt_text = f"""
    Please determine whether the following user question is related to searching for hotels.
    Answer with "Yes" or "No" only.

    User Question: "{question}"
    """
    response = llm.invoke(prompt_text)
    answer = response.content.strip().lower()
    print(f"Intent Classification Result: {answer.capitalize()}")
    return answer.startswith("y")

# Define application state
class State(TypedDict):
    question: str
    context: List[str]
    answer: str
    hotel_found: str  # Store the name of the found hotel if any

######################
# Integrated hotel search logic
######################

def load_data(file_path):
    """
    Loads the hotel dataset and cleans column names.
    """
    try:
        hotels_data = pd.read_csv(file_path, encoding="Windows-1252")
        hotels_data.columns = hotels_data.columns.str.strip()  # Clean column names
        print("--------------------------------------------------")
        print("Dataset loaded successfully!")
        print("Available columns in the dataset:")
        print(hotels_data.columns.tolist())
        print("--------------------------------------------------")
        return hotels_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def parse_user_input(user_input, column_names):
    """
    Uses LLM to extract filtering criteria and return fields.
    """
    one_shot_example = """
    Example User Input: "I want to find a hotel in London, with 3 star and wifi"
    Example Output: {
      "filters": {"cityName": "London", "HotelRating": "ThreeStar", "HotelFacilities": "Wifi"}
    }
    """

    prompt_text = f"""
    The dataset has the following columns: {column_names}
    Based on this, analyze the user's input and determine:
    1. The filtering criteria (e.g., "cityName": "New York").

    Provide your answer in JSON format like this:
    {{
      "filters": {{"column_name": "value", "another_column": "value"}}
    }}

    {one_shot_example}

    User Input: "{user_input}"
    """
    response = llm.invoke(prompt_text)
    response_content = response.content.strip()

    if "Output:" in response_content:
        response_content = response_content.split("Output:", 1)[1].strip()

    try:
        parsed_request = json.loads(response_content)
        return parsed_request
    except json.JSONDecodeError:
        print("--------------------------------------------------")
        print("Error parsing LLM response as JSON.")
        print("Please try rephrasing your query.")
        print("--------------------------------------------------")
        raise ValueError("LLM returned an invalid JSON response.")

def wrap_text_columns(df, columns_to_wrap, width=80):
    """
    Wrap text for specified columns to improve readability.
    """
    for col in columns_to_wrap:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: "\n".join(textwrap.wrap(x, width=width)))
    return df

def apply_filters(hotels_df, filters):
    """
    Apply filtering criteria to the dataset.
    """
    print("--------------------------------------------------")
    print("Applying filters to the dataset...")
    print("Current filters:", filters)
    filtered_df = hotels_df.copy()

    for column, value in filters.items():
        if column == "HotelFacilities" and isinstance(value, list):
            for condition in value:
                filtered_df = filtered_df[filtered_df[column].str.contains(condition, case=False, na=False)]
        else:
            filtered_df = filtered_df[filtered_df[column].str.contains(value, case=False, na=False)]

    result_fields = [
        "HotelName", "HotelRating", "Attractions", "Description",
        "HotelFacilities", "PhoneNumber", "HotelWebsiteUrl"
    ]
    result_fields = [f for f in result_fields if f in filtered_df.columns]
    result_df = filtered_df[result_fields]

    print(f"Number of results found: {len(result_df)}")

    columns_to_wrap = ["Description", "Attractions", "HotelFacilities"]
    result_df = wrap_text_columns(result_df, columns_to_wrap, width=80)

    if not result_df.empty:
        print("--------------------------------------------------")
        print("Sample of the matched results:")
        print(result_df.head().to_string(index=False))
    print("--------------------------------------------------")

    return result_df

def iterative_filtering(hotels_df, filters, column_names):
    """
    Iteratively refine results until one match remains.
    Maintains a record of at least one fallback hotel.
    Returns the final results (including if single or multiple hotels found).
    """
    if "HotelFacilities" in filters and not isinstance(filters["HotelFacilities"], list):
        filters["HotelFacilities"] = [filters["HotelFacilities"]]

    recorded_hotel = None

    while True:
        results = apply_filters(hotels_df, filters)

        if not results.empty:
            recorded_hotel = results.head(1)

        if results.empty:
            if recorded_hotel is not None and not recorded_hotel.empty:
                hotel_name = recorded_hotel.iloc[0].get("HotelName", "Unknown Hotel")
                print("No new results found. Returning the last recorded hotel:")
                print("--------------------------------------------------")
                print(f"The recommended hotel is: {hotel_name}")
                print("--------------------------------------------------")
                return recorded_hotel
            else:
                print("No results found and no fallback hotel available.")
                return None

        if len(results) == 1:
            # Final single hotel found
            hotel_name = results.iloc[0].get("HotelName", "Unknown Hotel")
            print("Final match found:")
            print("--------------------------------------------------")
            print(f"The recommended hotel is: {hotel_name}")
            print("--------------------------------------------------")
            return results

        # Multiple matches, ask for refinement
        print(f"Multiple matches found ({len(results)}). Please refine your search criteria.")
        user_input = input("Enter additional requirements (e.g., 'parking', 'free wifi', etc.):\n")
        additional_filters = parse_user_input(user_input, column_names).get("filters", {})
        if "HotelFacilities" in additional_filters:
            if "HotelFacilities" not in filters:
                filters["HotelFacilities"] = []
            filters["HotelFacilities"].append(additional_filters["HotelFacilities"])
        else:
            filters.update(additional_filters)

def run_hotel_search(file_path, user_query):
    """
    Runs the hotel search process using the user's original query.
    Returns the final search results (None, single DataFrame, or multiple).
    """
    hotels_data = load_data(file_path)
    if hotels_data is None:
        return None

    column_names = hotels_data.columns.tolist()
    initial_filters = parse_user_input(user_query, column_names).get("filters", {})
    results = iterative_filtering(hotels_data, initial_filters, column_names)
    return results


######################
# Original retrieval and generate logic
######################

def retrieve(state: State):
    """
    Determine if the user's query is hotel-related. If yes, run hotel search.
    Otherwise, do the existing retrieval logic.
    If a single hotel is found, store its info in state["context"] and hotel name in state["hotel_found"].
    """
    is_hotel_related = llm_intent_classification(state["question"])
    # Do not clear context here anymore, to maintain memory of previously found hotels
    if "context" not in state:
        state["context"] = []
    if "hotel_found" not in state:
        state["hotel_found"] = ""

    if is_hotel_related:
        print("System: Detected hotel-related query. Running hotel search...")
        file_path = r"G:\Code\Projects\GWU\24_FA\AML\Final_proj\hotels_sampled.csv"
        results = run_hotel_search(file_path, state["question"])

        if results is not None and len(results) == 1:
            hotel_info = results.iloc[0].to_dict()
            hotel_name = hotel_info.get("HotelName", "N/A")
            state["hotel_found"] = hotel_name
            context_str = (
                f"Hotel Name: {hotel_name}\n"
                f"Rating: {hotel_info.get('HotelRating','N/A')}\n"
                f"Attractions Nearby: {hotel_info.get('Attractions','N/A')}\n"
                f"Description: {hotel_info.get('Description','N/A')}\n"
                f"Facilities: {hotel_info.get('HotelFacilities','N/A')}\n"
                f"Phone: {hotel_info.get('PhoneNumber','N/A')}\n"
                f"Website: {hotel_info.get('HotelWebsiteUrl','N/A')}"
            )
            # Append the found hotel info to context if it's not already there
            if context_str not in state["context"]:
                state["context"].append(context_str)
        # If multiple hotels or None, no single hotel to record here.
    else:
        print("System: Detected non-hotel-related query. Searching attractions database...")
        # Add non-hotel info for context
        attraction_info = "Attraction Example: Central Park in New York, Open Hours: 6 AM - 1 AM, Admission: Free"
        if attraction_info not in state["context"]:
            state["context"].append(attraction_info)
    
    return state

def generate(state: State):
    """
    Generates an answer by combining the user question and retrieved context.
    If we have a single hotel in context, this will summarize that info.
    """
    if not state["context"]:
        state["answer"] = "I'm sorry, I couldn't find any relevant information."
        return state
    
    context_str = "\n".join(state["context"])
    prompt_text = f"""
    you are a travel agent,Based on the user's question and the following information, provide a helpful answer.

    User Question: "{state['question']}"
    Retrieved Information:
    {context_str}

    Answer:
    """
    response = llm.invoke(prompt_text)
    state["answer"] = response.content.strip()
    return state

def chat():
    """
    Simulates a continuous conversation loop.
    """
    print("Welcome to the Travel Assistant! Type 'exit' or 'quit' to end the conversation.")
    # Add hotel_announcement_done to track if we've printed the congratulatory message
    state = State(question="", context=[], answer="", hotel_found="")
    state["hotel_announcement_done"] = False

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Travel Assistant: Thank you for using the Travel Assistant. Goodbye!")
            break
        
        state["question"] = user_input
        
        # Retrieve relevant information based on intent
        previous_hotel_found = state.get("hotel_found", "")
        state = retrieve(state)
        current_hotel_found = state.get("hotel_found", "")

        # Check if we have just found a hotel for the first time
        if current_hotel_found and not state["hotel_announcement_done"]:
            hotel_name = current_hotel_found
            print(f"Congrats, we found you a hotel: {hotel_name}, what else do you want to know?")
            # Mark that we've shown the announcement
            state["hotel_announcement_done"] = True
            # Do not generate LLM response this turn, as per requirement
        else:
            # If no new hotel was found this turn, or we've already shown the announcement
            # Now handle subsequent queries.

            # If the user's question is not hotel-related but we have known hotel info
            # (You have mentioned "will be implemented" for other info logic)
            # So here's where you can print additional system messages or process non-hotel info.
            is_hotel_related = llm_intent_classification(state["question"])
            if not is_hotel_related:
                # Non-hotel query
                print("System: Detected non-hotel-related query. Searching attractions database...")
                # If we have known hotel info in context, we can still use it
                # The user wants the LLM to respond with known hotel info plus additional logic.
                # Since we still have the hotel in context (if found previously), we can generate a response.
            
            if state["context"]:
                state = generate(state)
                print(f"Travel Assistant: {state['answer']}\n")

        # Reset answer only (keep context and hotel_found for future queries)
        state["answer"] = ""



if __name__ == "__main__":
    chat()
