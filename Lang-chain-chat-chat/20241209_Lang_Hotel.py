import pandas as pd
from langchain_openai import ChatOpenAI
import json
import textwrap

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

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
    Cleans the response to ensure valid JSON parsing.
    """
    # One-shot example for LLM
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

    # Attempt to clean the response by removing any extra formatting
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

import pandas as pd
import textwrap

def wrap_text_columns(df, columns_to_wrap, width=80):
    """
    Wrap text for specified columns in the DataFrame so that 
    they do not produce overly long lines.
    """
    for col in columns_to_wrap:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: "\n".join(textwrap.wrap(x, width=width)))
    return df

def apply_filters(hotels_df, filters):
    """
    Filters the dataset based on criteria provided by the LLM.
    Supports multiple conditions for 'HotelFacilities'.
    """
    print("--------------------------------------------------")
    print("Applying filters to the dataset...")
    print("Current filters:", filters)
    filtered_df = hotels_df.copy()

    for column, value in filters.items():
        if column == "HotelFacilities" and isinstance(value, list):
            # Apply all conditions in the 'HotelFacilities' list
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

    # Wrap text in certain columns to improve readability
    columns_to_wrap = ["Description", "Attractions", "HotelFacilities"]
    result_df = wrap_text_columns(result_df, columns_to_wrap, width=80)

    if not result_df.empty:
        print("--------------------------------------------------")
        print("Sample of the matched results:")
        # Print only the first few rows for readability
        print(result_df.head().to_string(index=False))
    print("--------------------------------------------------")

    return result_df


def iterative_filtering(hotels_df, filters, column_names):
    """
    Iteratively refines results until only one match remains or the user stops.
    Keeps a record of at least one hotel as fallback.
    """
    # Initialize 'HotelFacilities' filter as a list if not present
    if "HotelFacilities" in filters and not isinstance(filters["HotelFacilities"], list):
        filters["HotelFacilities"] = [filters["HotelFacilities"]]

    recorded_hotel = None  # Keep track of a fallback hotel

    while True:
        results = apply_filters(hotels_df, filters)

        if not results.empty:
            # Update recorded_hotel with at least one fallback option
            # Take the first result as fallback record
            recorded_hotel = results.head(1)

        if results.empty:
            # No matches found; fallback to recorded_hotel if available
            if recorded_hotel is not None and not recorded_hotel.empty:
                print("No new results found. Returning the last recorded hotel:")
                print("--------------------------------------------------")
                print(recorded_hotel.to_string(index=False))
                print("--------------------------------------------------")
                return recorded_hotel
            else:
                # No fallback available
                print("No results found and no fallback hotel available.")
                return None

        if len(results) == 1:
            # Only one match: this becomes the final recorded hotel
            recorded_hotel = results
            print("Final match found:")
            print("--------------------------------------------------")
            print(results.to_string(index=False))
            print("--------------------------------------------------")
            return results

        # If multiple results, ask user for additional filters
        print(f"Multiple matches found ({len(results)}). Please refine your search criteria.")
        user_input = input("Enter additional requirements (e.g., 'parking', 'free wifi', etc.):\n")

        additional_filters = parse_user_input(user_input, column_names).get("filters", {})

        # Update 'HotelFacilities' filter if applicable
        if "HotelFacilities" in additional_filters:
            if "HotelFacilities" not in filters:
                filters["HotelFacilities"] = []
            filters["HotelFacilities"].append(additional_filters["HotelFacilities"])
        else:
            filters.update(additional_filters)


def main():
    """
    Main function to load data, parse user input, iteratively filter data, and display results.
    Keeps a record of at least one hotel at all times.
    """
    file_path = r"G:\Code\Projects\GWU\24_FA\AML\Final_proj\hotels_sampled.csv"
    hotels_data = load_data(file_path)
    if hotels_data is not None:
        column_names = hotels_data.columns.tolist()

        # Get initial user input
        user_input = input("Enter your initial query (e.g., 'Find hotels in New York with FourStar rating'):\n")

        # Parse initial user input using LLM
        initial_filters = parse_user_input(user_input, column_names).get("filters", {})

        # Start iterative filtering process
        results = iterative_filtering(hotels_data, initial_filters, column_names)

        if results is not None and len(results) > 1:
            print("--------------------------------------------------")
            print("Final filtered results:")
            print(results.to_string(index=False))
            print("--------------------------------------------------")
        elif results is None or results.empty:
            print("No results could be found with the given criteria.")
        # If one result, it's already printed out in the iterative_filtering function.

if __name__ == "__main__":
    main()
