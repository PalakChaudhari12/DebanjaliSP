import os
import pickle
import warnings
from typing import List, Tuple
import chainlit as cl
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import pdfplumber
import fitz

warnings.filterwarnings("ignore")

CACHE_FILE = "chat_history.pkl"

# Ensure Google API Key is set
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"
if GOOGLE_API_KEY_ENV not in os.environ:
    os.environ[GOOGLE_API_KEY_ENV] = "AIzaSyBnYvh7gby5inBuVIMYZp-RzhRdFicd7R4"

def save_history_to_cache(history: List[Tuple[str, str]]):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(history, f)

def load_history_from_cache() -> List[Tuple[str, str]]:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return []

def clear_history_cache():
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

class DocumentProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.documents = PyPDFLoader(pdf_path).load()
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = None
        self.split_docs = None

    def process(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.split_docs = splitter.split_documents(self.documents)
        self.vector_store = Chroma.from_documents(self.split_docs, self.embeddings)

    def get_retriever(self, k: int = 3):
        return self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k})

class QueryHandler:
    def __init__(self, retriever, pdf_name: str):
        self.retriever = retriever
        self.pdf_name = pdf_name
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        self.history = load_history_from_cache()

    def add_to_history(self, query: str, answer: str):
        self.history.append((query, answer))
        save_history_to_cache(self.history)

    def build_context(self) -> str:
        return "\n".join([f"Q: {q}\nA: {a}" for q, a in self.history])

    def clear_page_highlights(self, page):
        pdf_doc = fitz.open(self.pdf_name)
        annotations = page.annots()
        if annotations:
            for annot in annotations:
                if annot.type[0] == 8:  # Highlight annotation type
                    page.delete_annot(annot)
        pdf_doc.saveIncr()
        pdf_doc.close()

    def highlight_text(self, page_number: int, text: str):
        pdf_doc = fitz.open(self.pdf_name)
        page = pdf_doc[page_number - 1]  # Pages are 0-indexed in PyMuPDF
        self.clear_page_highlights(page)

        text_instances = page.search_for(text)
        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)
            highlight.update()

        pdf_doc.saveIncr()
        pdf_doc.close()

    async def generate_answer(self, query: str) -> dict:
        try:
        # Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(query)
            relevant_text = "\n".join(doc.page_content for doc in docs)

        # Instructions for the assistant
            instructions = (
            "You are an AI assistant tasked with answering questions based on the provided document. "
            "Your responses should be concise, accurate, and supported by the content from the document."
            )

        # Build conversation context
            context = self.build_context()
            if context:
                prompt = (
                    f"{instructions}\n\n"
                    f"Here is the conversation history to help you understand the context:\n"
                    f"{context}\n\n"
                    f"Now, answer the following query:\n{query}\n\n"
                    f"Based on the following relevant text from the document:\n{relevant_text}"
                )
            else:
                prompt = (
                    f"{instructions}\n\n"
                    f"Answer the following query:\n{query}\n\n"
                    f"Based on the following relevant text from the document:\n{relevant_text}"
                )

        # Create messages for LLM
            system_message = SystemMessage(content="You are an AI assistant answering questions based on a document.")
            user_message = HumanMessage(content=prompt)

        # Get the answer from the language model
            response = await self.llm.apredict_messages([system_message, user_message])
            answer = response.content.strip()

        # Highlight text and create citations
            citations = []
            for doc in docs:
                page_num = doc.metadata.get("page")
                if page_num is not None:
                    self.highlight_text(page_num + 1, doc.page_content)
                    citations.append({"page": page_num + 1, "citation": f"Page {page_num + 1}"})

        # Save to history
            self.add_to_history(query, answer)
            return {"query": query, "answer": answer, "citations": citations}
        except Exception as e:
            return {"query": query, "answer": f"Error generating answer: {str(e)}", "citations": []}

welcome_message = """Welcome to the PDF Chatbot! To get started:
1. Upload a PDF file
2. Ask a question about the file
3. To see history type !view_history
4. To clear history type !clear_history
"""
async def initial(content):
    print("before")
    await cl.Message(content=content).send()
    print("after")

@cl.on_chat_start
async def handle_chat_start():
    files = None
    while not files:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()
        await cl.Message(content="Processing!!!").send()
    file = files[0]
    await cl.Message(
        content=f"`{file.name}` processing !"
    ).send()
    print("-----------------------------------------------")
    processor = DocumentProcessor(file.path)
    processor.process()
    retriever = processor.get_retriever()

    query_handler = QueryHandler(retriever, file.name)
    cl.user_session.set("query_handler", query_handler)
    if file:
        await cl.Message(content=f"`{file.name}` uploaded, it contains").send()
    await cl.Message(content = "Processing complete. You can now ask questions!").send()  # Update the previously sent message


@cl.on_message
async def handle_message(message: cl.Message):
    query_handler = cl.user_session.get("query_handler")
    
    if message.content.strip().lower() == "!view_history":
        # Display chat history
        history = load_history_from_cache()
        if history:
            history_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in history])
            await cl.Message(content=f"Chat History:\n\n{history_text}").send()
        else:
            await cl.Message(content="No chat history found.").send()
        return

    if message.content.strip().lower() == "!clear_history":
        # Clear chat history
        clear_history_cache()
        await cl.Message(content="Chat history cleared successfully.").send()
        return

    # Handle regular messages
    response = await query_handler.generate_answer(message.content)
    answer = response["answer"]
    citations = response["citations"]

    citation_texts = ""
    if citations:
        citation_texts = "\n\nCitations:\n" + "\n".join(
            f"[Page {c['page']}]({query_handler.pdf_name}#page={c['page']})"
            for c in citations
        )

    elements = [
        cl.Pdf(
            name=query_handler.pdf_name,
            display="inline",
            path=query_handler.pdf_name,
            page=citation["page"] - 1,  # Convert to 0-based indexing for the viewer
            link_text=f"Go to page {citation['page']}",
        )
        for citation in citations if citation["page"] is not None
    ]

    await cl.Message(content=f"{answer}{citation_texts}", elements=elements).send()