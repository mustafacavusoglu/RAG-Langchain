# RAG Chat Application with AI Assistant

This is a chat application developed using Streamlit and Langchain Python libraries. The primary functionality of the application is to facilitate interaction between users and an artificial intelligence assistant.

## Overview

The key features provided by the application include:

1. **AI Assistant**: The application incorporates an artificial intelligence assistant powered by OpenAI's GPT-3.5 model. This assistant can comprehend user queries and generate responses based on a specific set of documents.

2. **Document Loading**: By loading documents from a specified directory, the application utilizes them to generate responses for the AI assistant. These documents serve as the basis for the assistant's answers.

3. **Chat Interface**: Developed using Streamlit, the application offers a chat interface where users can input their queries and receive responses from the AI assistant. The chat interface provides an interactive and user-friendly experience.

## Installation

To run the application, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/RAG-Langchain.git
```
2. Install the required dependencies using pip and the provided requirements.txt file: 
```bash
pip install -r requirements.txt
```
3. Navigate to the project directory:
```bash
cd RAG-Langchain
```

4. Create a .env file in the project directory and add your OpenAI API key:
```plaintext
OPEN_AI_KEY=your_openai_api_key_here
```
5. Run application:
```bash
streamlit run app.py
```


## Acknowledgements

This project makes use of various libraries and technologies, including Streamlit, Langchain, and OpenAI. We acknowledge the contributions of the developers and maintainers of these libraries.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



