Certainly! Below is a comprehensive **README** for your GitHub project. This README is structured to provide clear and detailed information about your project, making it easy for others to understand, install, and use your Travel Agent LLM system.

---

# A Travel Agent LLM

**Integrating LangChain, Knowledge Graphs, and Retrieval-Augmented Generation for Hotel and Attraction Recommendations**

![Project Banner](https://your-image-link.com/banner.png) <!-- Replace with an actual image link if available -->

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Introduction

Welcome to **A Travel Agent LLM**, an advanced conversational travel assistant designed to help users find hotels, attractions, and transportation options through natural language queries. By integrating a Large Language Model (LLM), LangChain, a custom Knowledge Graph (KG), and Retrieval-Augmented Generation (RAG), this system provides intelligent and contextually relevant recommendations to enhance user travel planning experiences.

## Features

- **Intent Classification:** Distinguishes between hotel-related and non-hotel-related user queries.
- **Hotel Recommendations:** Utilizes rule-based filtering on a comprehensive hotel dataset to provide accurate hotel suggestions.
- **Attraction and Transportation Suggestions:** Employs semantic embedding-based retrieval from a custom Knowledge Graph to recommend attractions and transport options.
- **Conversational Memory:** Maintains context for follow-up queries, ensuring a seamless conversational experience.
- **Retrieval-Augmented Generation:** Enhances response generation by integrating retrieved data and contextual information.
- **Scalable Knowledge Graph:** Initially based on New York City, the Knowledge Graph can be extended to include additional destinations.

## Technologies Used

- **Language Model:** [OpenAI GPT-3.5](https://openai.com/)
- **Framework:** [LangChain](https://python.langchain.com/)
- **Knowledge Graph:** Custom-built using [NetworkX](https://networkx.org/)
- **Vector Store:** [FAISS](https://faiss.ai/)
- **Data Sources:** [Kaggle Hotel Dataset](https://www.kaggle.com/datasets)
- **Programming Language:** Python
- **Visualization:** [TikZ](https://tikz.dev/)
- **Documentation:** LaTeX

## Architecture

The system architecture integrates multiple components to deliver a robust travel assistant:

1. **User Interface:** Accepts natural language queries from users.
2. **Intent Classification:** Utilizes an LLM to categorize queries as hotel-related or non-hotel-related.
3. **Data Retrieval:**
   - **Hotel Queries:** Applies rule-based filters on the hotel CSV dataset to find relevant hotels.
   - **Non-Hotel Queries:** Performs semantic searches on the Knowledge Graph using embeddings and FAISS for similarity search.
4. **Response Generation:** Combines retrieved data and context to generate coherent responses using the LLM.
5. **Conversational Memory:** Maintains context for handling follow-up queries seamlessly.


## Installation

Follow these steps to set up the project locally:


### Prepare the Data

- **Hotel Dataset:** Download the hotel dataset from [Kaggle](https://www.kaggle.com/datasets) and place the CSV file in the `data/` directory.
- **Knowledge Graph:** Ensure the Knowledge Graph data is available in the `data/kg/` directory. You may need to preprocess or extend the KG based on your requirements.

## Usage

Run the main application to start the travel assistant.


## Evaluation

The system was evaluated based on several performance metrics:

### Categorical Metrics Comparison

| **Metric**                 | **With RAG**                              | **Without RAG**                  |
|----------------------------|-------------------------------------------|----------------------------------|
| **Hotels Provided**        | 3,994                                     | Numerous (exact number not provided) |
| **4-Star Hotels**          | 490                                       | Numerous                          |
| **5-Star Hotels**          | 89                                        | Numerous                          |
| **Correct Contact Info**   | Yes                                       | No, Incorrect, Made-up            |
| **Info on Getting There**  | Yes                                       | Yes                               |
| **Response Cost**          | High                                      | Medium                            |
| **Response Time**          | Medium                                    | Low                               |

### Model Performance Statistics (Last 7 Days)

| **Statistic**     | **Value**             |
|-------------------|-----------------------|
| **Run Count**     | 267                   |
| **Total Tokens**  | 80,001 / \$1.02       |
| **Median Tokens** | 221                   |
| **Error Rate**    | 2%                    |
| **% Streaming**   | 0%                    |
| **Latency**       | P50: 0.68s, P99: 10.13s |

**Analysis:**

- The integration of RAG significantly improved the accuracy of responses, especially for non-hotel queries.
- The system maintained a low error rate and acceptable latency, ensuring a reliable user experience.
- Response costs are higher with RAG integration, which is a trade-off for improved accuracy and relevance.


5. **Open a Pull Request**

Please ensure your code follows the project's coding standards and includes relevant tests.

## License

This project is licensed under the [Creative Commons CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Contact

**Shi Qiu**  
The George Washington University  
Email: [tusrau@gmail.com](mailto:tusrau@gmail.com)

For any inquiries or feedback, please reach out via email or open an issue on GitHub.

## Acknowledgements

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [FAISS Vector Store](https://faiss.ai/)
- [NetworkX](https://networkx.org/)
- [Kaggle Hotel Dataset](https://www.kaggle.com/datasets)
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [Knowledge Graphs](https://en.wikipedia.org/wiki/Knowledge_graph)
- [Lilian Weng's Agent Framework Overview](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
