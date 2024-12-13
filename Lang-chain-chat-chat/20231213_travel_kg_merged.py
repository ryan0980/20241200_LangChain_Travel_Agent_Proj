# Imports and Environment Setup
import os
from dotenv import load_dotenv
import pandas as pd
import json
import textwrap
from typing_extensions import List, TypedDict
from langchain_openai import ChatOpenAI

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 加载环境变量
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")

print("OPENAI_API_KEY:", "*****" if os.getenv("OPENAI_API_KEY") else "Not Set")

# Initialize ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

# 定义 State TypedDict
class State(TypedDict):
    question: str
    context: List[str]
    answer: str
    hotel_found: str  # 用于存储找到的单一酒店的名称

###############################################
# 酒店搜索相关函数
###############################################

def llm_intent_classification_hotels(question: str) -> bool:
    """
    使用 LLM 来确定用户问题是否与搜索酒店相关。
    返回 True 如果是与酒店相关，否则 False。
    """
    prompt_text = f"""
    Please determine whether the following user question is related to searching for hotels.
    Answer with "Yes" or "No" only.

    User Question: "{question}"
    """
    response = llm.invoke(prompt_text)
    answer = response.content.strip().lower()
    print(f"Intent Classification (Hotel) Result: {answer.capitalize()}")
    return answer.startswith("y")

def load_data(file_path):
    """
    加载酒店数据集，并清理列名
    """
    try:
        hotels_data = pd.read_csv(file_path, encoding="Windows-1252")
        hotels_data.columns = hotels_data.columns.str.strip()  # 清洗列名
        print("--------------------------------------------------")
        print("Dataset loaded successfully!")
        print("Available columns in the dataset:")
        print(hotels_data.columns.tolist())
        print("--------------------------------------------------")
        return hotels_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def parse_user_input_for_hotels(user_input, column_names):
    """
    使用LLM从用户输入中提取筛选条件，并返回fields
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
        print("Error parsing LLM response as JSON for hotel filters.")
        print("Please try rephrasing your query.")
        print("--------------------------------------------------")
        raise ValueError("LLM returned an invalid JSON response.")

def wrap_text_columns(df, columns_to_wrap, width=80):
    """
    包装文本列用于更好的可读性
    """
    for col in columns_to_wrap:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: "\n".join(textwrap.wrap(x, width=width)))
    return df

def apply_filters(hotels_df, filters):
    """
    根据筛选条件对数据集进行过滤
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
    迭代式地缩小筛选范围，直到只剩一家酒店或无结果。
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
            # 只剩一家酒店
            hotel_name = results.iloc[0].get("HotelName", "Unknown Hotel")
            print("Final match found:")
            print("--------------------------------------------------")
            print(f"The recommended hotel is: {hotel_name}")
            print("--------------------------------------------------")
            return results

        # 如果还有多家酒店，让用户进一步添加条件
        print(f"Multiple matches found ({len(results)}). Please refine your search criteria.")
        user_input = input("Enter additional requirements (e.g., 'parking', 'free wifi', etc.):\n")
        additional_filters = parse_user_input_for_hotels(user_input, column_names).get("filters", {})
        if "HotelFacilities" in additional_filters:
            if "HotelFacilities" not in filters:
                filters["HotelFacilities"] = []
            filters["HotelFacilities"].append(additional_filters["HotelFacilities"])
        else:
            filters.update(additional_filters)

def run_hotel_search(file_path, user_query):
    """
    根据用户查询执行酒店搜索，返回最终结果
    """
    hotels_data = load_data(file_path)
    if hotels_data is None:
        return None

    column_names = hotels_data.columns.tolist()
    initial_filters = parse_user_input_for_hotels(user_query, column_names).get("filters", {})
    results = iterative_filtering(hotels_data, initial_filters, column_names)
    return results

###############################################
# 知识图谱相关函数
###############################################

class KGState(TypedDict):
    question: str
    context: List[str]
    answer: str

class KnowledgeGraph:
    def __init__(self):
        self.G = nx.Graph()
        self._initialize_graph()

    def _initialize_graph(self):
        # 添加城市
        self.G.add_node("New York City", type="City", country="USA")

        # 添加交通节点
        transport_places = [
            {"name": "Grand Central Terminal", "type": "TransportPlace", "transport_type": "Train Station"},
            {"name": "JFK Airport", "type": "TransportPlace", "transport_type": "Airport"},
            {"name": "Port Authority Bus Terminal", "type": "TransportPlace", "transport_type": "Bus Terminal"},
            {"name": "Times Square Subway Station", "type": "TransportPlace", "transport_type": "Subway Station"},
        ]

        for transport in transport_places:
            self.G.add_node(transport["name"], type=transport["type"], transport_type=transport["transport_type"])
            self.G.add_edge("New York City", transport["name"], relationship="HAS_TRANSPORT")

        # 添加景点
        attractions = [
            {"name": "Central Park", "category": "Park"},
            {"name": "Times Square", "category": "Entertainment"},
            {"name": "Statue of Liberty", "category": "Monument"},
            {"name": "Empire State Building", "category": "Skyscraper"},
            {"name": "Metropolitan Museum of Art", "category": "Museum"},
            {"name": "Brooklyn Bridge", "category": "Bridge"},
        ]

        for attraction in attractions:
            self.G.add_node(attraction["name"], type="Attraction", category=attraction["category"])
            self.G.add_edge("New York City", attraction["name"], relationship="HAS_ATTRACTION")

        # 添加酒店
        hotels = [
            {"name": "Collective Governors Island", "type": "Hotel"},
        ]

        for hotel in hotels:
            self.G.add_node(hotel["name"], type=hotel["type"])
            self.G.add_edge("New York City", hotel["name"], relationship="HAS_HOTEL")
            self.G.add_edge(hotel["name"], "Statue of Liberty", relationship="NEAR")

        # 添加景点与交通的关系
        attraction_transport = {
            "Central Park": ["Times Square Subway Station"],
            "Times Square": ["Times Square Subway Station"],
            "Statue of Liberty": ["JFK Airport"],
            "Empire State Building": ["Times Square Subway Station"],
            "Metropolitan Museum of Art": ["Grand Central Terminal"],
            "Brooklyn Bridge": ["Port Authority Bus Terminal"],
        }

        for attraction, transports in attraction_transport.items():
            for transport in transports:
                self.G.add_edge(attraction, transport, relationship="NEAR")

    def find_attractions(self, city):
        return [
            node for node, attr in self.G.nodes(data=True)
            if attr.get("type") == "Attraction" and
            self.G.has_edge(city, node) and
            self.G.edges[city, node].get("relationship") == "HAS_ATTRACTION"
        ]

    def find_transport_places(self, city):
        return [
            node for node, attr in self.G.nodes(data=True)
            if attr.get("type") == "TransportPlace" and self.G.has_edge(city, node, relationship="HAS_TRANSPORT")
        ]

    def find_transport_near_attraction(self, attraction):
        return [
            neighbor for neighbor in self.G.neighbors(attraction)
            if self.G.edges[attraction, neighbor]["relationship"] == "NEAR"
        ]

    def find_attractions_near_transport(self, transport_place):
        return [
            neighbor for neighbor in self.G.neighbors(transport_place)
            if self.G.edges[transport_place, neighbor]["relationship"] == "NEAR"
        ]

kg = KnowledgeGraph()

def visualize_graph(kg: KnowledgeGraph):
    G = kg.G
    plt.figure(figsize=(15, 10))

    # 节点颜色
    node_colors = []
    for node, attr in G.nodes(data=True):
        if attr["type"] == "City":
            node_colors.append("lightblue")
        elif attr["type"] == "Attraction":
            node_colors.append("lightcoral")
        elif attr["type"] == "TransportPlace":
            node_colors.append("gold")
        elif attr["type"] == "Hotel":
            node_colors.append("lightgreen")
        else:
            node_colors.append("grey")

    # 边颜色
    edge_colors = []
    for u, v, data in G.edges(data=True):
        relationship = data.get("relationship", "")
        if relationship == "HAS_ATTRACTION":
            edge_colors.append("red")
        elif relationship == "HAS_TRANSPORT":
            edge_colors.append("blue")
        elif relationship == "HAS_HOTEL":
            edge_colors.append("purple")
        elif relationship == "NEAR":
            edge_colors.append("green")
        else:
            edge_colors.append("black")

    pos = nx.spring_layout(G, k=0.5, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    node_legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='City'),
        Patch(facecolor='lightcoral', edgecolor='black', label='Attraction'),
        Patch(facecolor='gold', edgecolor='black', label='Transport Place'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Hotel'),
    ]

    edge_legend_elements = [
        Patch(facecolor='red', edgecolor='red', label='HAS_ATTRACTION'),
        Patch(facecolor='blue', edgecolor='blue', label='HAS_TRANSPORT'),
        Patch(facecolor='purple', edgecolor='purple', label='HAS_HOTEL'),
        Patch(facecolor='green', edgecolor='green', label='NEAR'),
    ]

    plt.legend(handles=node_legend_elements + edge_legend_elements, loc='upper right')
    plt.title("New York City Tourist Attractions, Transport, and Hotels Knowledge Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

###############################################
# 基于KG的简单解析与检索
###############################################

def kg_llm_intent_classification(question: str) -> str:
    question_lower = question.lower()
    if "attractions" in question_lower and "in" in question_lower:
        return "attractions_in_city"
    elif any(phrase in question_lower for phrase in ["how to get to", "how can i get to", "transport to"]):
        return "transport_to_attraction"
    elif any(phrase in question_lower for phrase in ["transport places", "transport options", "how to move around"]):
        return "transport_places_in_city"
    else:
        return "unknown"

def parse_user_input_for_kg(user_input, intent):
    filters = {}
    words = user_input.lower().split()

    if intent == "attractions_in_city":
        if "new york city" in user_input.lower():
            filters["city"] = "New York City"

    elif intent == "transport_to_attraction":
        for attraction in kg.find_attractions("New York City"):
            if attraction.lower() in user_input.lower():
                filters["attraction"] = attraction
                break

    elif intent == "transport_places_in_city":
        if "new york city" in user_input.lower():
            filters["city"] = "New York City"

    return filters

def kg_retrieve(state: State, kg: KnowledgeGraph):
    intent = kg_llm_intent_classification(state["question"])
    filters = parse_user_input_for_kg(state["question"], intent)

    if intent == "attractions_in_city":
        city = filters.get("city", "New York City")
        attractions = kg.find_attractions(city)
        if attractions:
            context = f"Attractions in {city}: {', '.join(attractions)}."
            state["context"].append(context)
        else:
            state["context"].append(f"No attractions found in {city}.")

    elif intent == "transport_to_attraction":
        attraction = filters.get("attraction", "")
        if attraction:
            transports = kg.find_transport_near_attraction(attraction)
            if transports:
                context = f"Transport options to {attraction}: {', '.join(transports)}."
                state["context"].append(context)
            else:
                state["context"].append(f"No transport options found near {attraction}.")
        else:
            state["context"].append("Attraction not recognized.")

    elif intent == "transport_places_in_city":
        city = filters.get("city", "New York City")
        transports = kg.find_transport_places(city)
        if transports:
            context = f"Transport places in {city}: {', '.join(transports)}."
            state["context"].append(context)
        else:
            state["context"].append(f"No transport places found in {city}.")
    else:
        state["context"].append("I'm sorry, I couldn't understand your query. Please try rephrasing.")

    return state

def generate_answer(state: State):
    if not state["context"]:
        state["answer"] = "I'm sorry, I couldn't find any relevant information."
        return state

    context_str = "\n".join(state["context"])
    prompt_text = f"""
    You are a travel assistant. Based on the user's question and the following information, provide a helpful answer.

    User Question: "{state['question']}"
    Retrieved Information:
    {context_str}

    Answer:
    """
    response = llm.invoke(prompt_text)
    state["answer"] = response.content.strip()
    return state

###############################################
# 最终的聊天逻辑
###############################################

def unified_retrieve(state: State):
    """
    使用LLM先判断是否与酒店搜索相关，如果是则调用酒店逻辑，否则调用KG逻辑。
    如果找到唯一的酒店，把酒店信息加入context。
    """
    is_hotel_related = llm_intent_classification_hotels(state["question"])
    if "context" not in state:
        state["context"] = []
    if "hotel_found" not in state:
        state["hotel_found"] = ""

    if is_hotel_related:
        print("System: Detected hotel-related query. Running hotel search...")
        #file_path = r"G:\Code\Projects\GWU\24_FA\AML\Final_proj\hotels_sampled.csv" # 请根据实际路径修改
        file_path = r"G:\Code\Projects\GWU\24_FA\AML\Final_proj\hotels.csv"
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
            if context_str not in state["context"]:
                state["context"].append(context_str)
    else:
        print("System: Detected non-hotel-related query. Using Knowledge Graph...")
        # 使用KG来回答
        kg_retrieve(state, kg)

    return state

def chat():
    print("Welcome to the Travel Assistant! Type 'exit' or 'quit' to end the conversation.")
    state = State(question="", context=[], answer="", hotel_found="")
    state["hotel_announcement_done"] = False

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Travel Assistant: Thank you for using the Travel Assistant. Goodbye!")
            break

        state["question"] = user_input
        previous_hotel_found = state.get("hotel_found", "")
        state = unified_retrieve(state)
        current_hotel_found = state.get("hotel_found", "")

        # 如果刚刚找到酒店且还没公告过
        if current_hotel_found and not state["hotel_announcement_done"]:
            hotel_name = current_hotel_found
            print(f"Congrats, we found you a hotel: {hotel_name}, what else do you want to know?")
            state["hotel_announcement_done"] = True
            # 不生成回答，等待用户下一个问题
        else:
            # 非酒店查询时直接生成答案
            # 或者用户在找到酒店后继续提问其它问题
            state = generate_answer(state)
            print(f"Travel Assistant: {state['answer']}\n")

        state["answer"] = ""

if __name__ == "__main__":
    visualize_graph(kg)
    chat()
