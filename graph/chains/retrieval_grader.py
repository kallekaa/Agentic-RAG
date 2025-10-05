from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GradeDocument(BaseModel):
    """Binary score for relevance of a document to a question."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no' only.")

structured_llm_grader = llm.with_structured_output(GradeDocument)

system = """You are a helpful assistant that grades the relevance of a document to a question.
You will be given a question and a document. Give a binary score 'yes' or 'no' if the document is relevant to the question. 
Answer only with 'yes' or 'no'.\n"""

grade_prompt = ChatPromptTemplate.from_messages(
    [("system", system),
     ("human", "Retrived document: {document}\nQuestion: {question}\nIs the document relevant to the question? 'yes' or 'no'?\n")
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
