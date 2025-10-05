
def test_foo():
    assert True


from dotenv import load_dotenv
load_dotenv()

from graph.chains.retrieval_grader import retrieval_grader, GradeDocument
from ingestion import retriever

def test_retrieval_grader_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocument = retrieval_grader.invoke(
        {"document": doc_txt, "question": question}
    )

    assert res.binary_score.lower() == "yes"


def test_retrieval_grader_should_fail() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocument = retrieval_grader.invoke(
        {"document": doc_txt, "question": question}
    )

    assert res.binary_score.lower() == "no"



