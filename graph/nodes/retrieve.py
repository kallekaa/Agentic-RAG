from typing import Any, Dict, List, Optional

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("Retrieving relevant documents...")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
