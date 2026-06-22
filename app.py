import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Zyro Dynamics HR Help Desk")
st.write("Ask questions about Zyro Dynamics HR policies, benefits, leave, payroll, attendance, and employee guidelines.")

def get_secret(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)

LLM_PROVIDER = get_secret("LLM_PROVIDER", "groq")
LLM_MODEL = get_secret("LLM_MODEL", "llama-3.1-8b-instant")

if LLM_PROVIDER == "groq":
    os.environ["GROQ_API_KEY"] = get_secret("GROQ_API_KEY", "")
elif LLM_PROVIDER == "gemini":
    os.environ["GOOGLE_API_KEY"] = get_secret("GOOGLE_API_KEY", "")
elif LLM_PROVIDER == "openai":
    os.environ["OPENAI_API_KEY"] = get_secret("OPENAI_API_KEY", "")

CORPUS_PATHS = [
    "./corpus",
    "./zyro-dynamics-hr-corpus",
    "/kaggle/input/zyro-dynamics-hr-corpus",
    "/kaggle/input"
]

@st.cache_resource
def load_llm():
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=LLM_MODEL,
            temperature=0.1,
            max_tokens=512
        )

    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.1,
            max_output_tokens=512
        )

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.1,
            max_tokens=512
        )

    raise ValueError("Unsupported LLM_PROVIDER. Use groq, gemini, or openai.")

@st.cache_resource
def build_retriever():
    pdf_root = None

    for path in CORPUS_PATHS:
        if os.path.exists(path):
            pdf_files = list(Path(path).rglob("*.pdf"))
            if pdf_files:
                pdf_root = str(Path(pdf_files[0]).parent)
                break

    if pdf_root is None:
        raise FileNotFoundError(
            "No PDF files found. Add HR policy PDFs inside a folder named 'corpus'."
        )

    loader = PyPDFDirectoryLoader(pdf_root)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

def format_docs(docs):
    formatted = []

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", None)
        page_text = f", page {page + 1}" if isinstance(page, int) else ""
        formatted.append(
            f"[Document {i}] Source: {source}{page_text}\n{doc.page_content}"
        )

    return "\n\n".join(formatted)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Zyro Dynamics HR Help Desk Assistant.

Answer employee HR questions using ONLY the provided HR policy context.

Rules:
1. If the answer is present in the context, answer clearly and directly.
2. If exact details are not available, say that the policy documents do not contain enough information.
3. Do not make up HR rules, dates, numbers, benefits, salary details, leave counts, or compliance rules.
4. Keep answers simple, professional, and employee-friendly."""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}

Answer:"""
    )
])

OOS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Classify the question for an HR Help Desk.

Return only IN_SCOPE if it is about HR policies, leave, payroll, benefits, attendance, workplace conduct, onboarding, offboarding, reimbursements, performance, holidays, or employee rules.

Return only OUT_OF_SCOPE otherwise."""
    ),
    ("human", "Question: {question}")
])

REFUSAL_MESSAGE = (
    "I can only answer questions related to Zyro Dynamics HR policies and employee guidelines. "
    "Please ask about HR policies, leave, payroll, benefits, attendance, workplace conduct, or company employee rules."
)

HR_KEYWORDS = [
    "hr", "policy", "leave", "salary", "payroll", "benefit", "attendance",
    "employee", "manager", "holiday", "reimbursement", "travel", "remote",
    "work from home", "wfh", "onboarding", "offboarding", "performance",
    "appraisal", "probation", "resignation", "notice period", "training",
    "conduct", "compliance", "insurance", "medical", "claim", "expense",
    "overtime", "timesheet", "workplace", "promotion", "increment"
]

@st.cache_resource
def load_chains():
    llm = load_llm()
    retriever = build_retriever()
    rag_chain = RAG_PROMPT | llm | StrOutputParser()
    guardrail_chain = OOS_PROMPT | llm | StrOutputParser()
    return llm, retriever, rag_chain, guardrail_chain

try:
    llm, retriever, rag_chain, guardrail_chain = load_chains()
    st.success("HR policy assistant is ready.")
except Exception as e:
    st.error(f"Setup error: {e}")
    st.stop()

def ask_bot(question: str):
    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a valid HR policy question.",
            "sources": []
        }

    lowered = question.lower()
    keyword_match = any(keyword in lowered for keyword in HR_KEYWORDS)

    if not keyword_match:
        try:
            scope = guardrail_chain.invoke({"question": question}).strip().upper()
        except Exception:
            scope = "IN_SCOPE"

        if "OUT_OF_SCOPE" in scope and "IN_SCOPE" not in scope:
            return {
                "answer": REFUSAL_MESSAGE,
                "sources": []
            }

    docs = retriever.invoke(question)
    context = format_docs(docs)

    answer = rag_chain.invoke({
        "context": context,
        "question": question
    }).strip()

    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", None)
        page_num = page + 1 if isinstance(page, int) else None

        item = {
            "source": source,
            "page": page_num
        }

        if item not in sources:
            sources.append(item)

    return {
        "answer": answer,
        "sources": sources
    }

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask an HR policy question...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            result = ask_bot(question)
            st.markdown(result["answer"])

            if result.get("sources"):
                with st.expander("Sources"):
                    for src in result["sources"]:
                        page_text = f", page {src['page']}" if src.get("page") else ""
                        st.write(f"- {src['source']}{page_text}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })