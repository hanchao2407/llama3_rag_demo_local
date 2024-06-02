import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function
import time


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    start_time_total = time.time()  # Record the start time
    start_time_sim = time.time()
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    print(results)
    end_time_sim = time.time()
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
      # Record the end time

    # print(prompt)
     # Output the processing time

    # return
    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    end_time_total = time.time()
    processing_time_sim = end_time_sim - start_time_sim  # Calculate the processing time
    print(f"Similarity Search Processing time: {processing_time_sim:.2f} seconds") 
    processing_time_total = end_time_total - start_time_total  # Calculate the processing time
    print(f"Total Processing time: {processing_time_total:.2f} seconds") 
    return response_text


if __name__ == "__main__":
    main()
