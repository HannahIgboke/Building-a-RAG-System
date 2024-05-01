# 👨‍💻Building a RAG System

<p align="center">
    <img width="500" src="https://github.com/HannahIgboke/Building-a-RAG-System/blob/main/RAG.png" alt="RAG">
</p>


As we know that LLMs like Gemini lack the company specific information. But this latest information is available via PDFs, Text Files, etc... Now if we can connect our LLM with these sources, we can build a much better application.


Using LangChain framework, I built a  Retrieval Augmented Generation (RAG) system that can utilize the power of LLM like Gemini 1.5 Pro to answer questions on the “Leave No Context Behind” paper published by Google on 10th April 2024. In this process, external data(i.e. the Leave No Context Behind Paper) is retrieved and then passed to the LLM during the generation step.

You can find the paper [here](https://arxiv.org/pdf/2404.07143.pdf).

# 🛠 Tech stack
- Langchain
- ChromaDB
- Streamlit


Find the complete code implementation [here](https://github.com/HannahIgboke/Building-a-RAG-System/blob/main/RAG_app.py).
