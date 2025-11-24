from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

# Google GenAI Integration Imports
from haystack_integrations.components.embedders.google_genai import GoogleGenAIDocumentEmbedder, GoogleGenAITextEmbedder
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

# Environment Setup for Google GenAI
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv("./.env")  # Load environment variables from .env file

class APIConfig(BaseSettings):
    GEMINI_API_KEY: str
    GEMINI_MODEL_NAME: str
    GEMINI_EMBEDDING_MODEL_NAME: str

config = APIConfig()

# 1. Setup Document Store
document_store = InMemoryDocumentStore()

# 2. Build the Indexing Pipeline (Audio -> Google Embeddings -> Vector DB)
indexing_pipeline = Pipeline()

# Preprocessing
cleaner = DocumentCleaner()
splitter = DocumentSplitter(split_by="word", split_length=200, split_overlap=20)

# Google Embedder for Documents
# using 'models/text-embedding-004' which is efficient for retrieval
doc_embedder = GoogleGenAIDocumentEmbedder(
    model=config.GEMINI_EMBEDDING_MODEL_NAME,
    api_key=Secret.from_token(config.GEMINI_API_KEY)
)

writer = DocumentWriter(document_store=document_store)

# Note: Transcriber component removed since we're using mock data
# In production, add a proper transcriber component here

indexing_pipeline.add_component("cleaner", cleaner)
indexing_pipeline.add_component("splitter", splitter)
indexing_pipeline.add_component("doc_embedder", doc_embedder)
indexing_pipeline.add_component("writer", writer)

indexing_pipeline.connect("cleaner", "splitter")
indexing_pipeline.connect("splitter", "doc_embedder")
indexing_pipeline.connect("doc_embedder", "writer")

# --- MOCKING DATA (To simulate audio transcription) ---
# In a real scenario, you would run: indexing_pipeline.run({"transcriber": {"sources": ["call.mp3"]}})
print("Indexing Data...")

mock_transcript = """
Sales Rep: Hi, thanks for jumping on the call. This is Mike from CloudScale.
Lead: Hi Mike. We're currently evaluating vendors for our data pipeline.
Sales Rep: Great. What are your main blockers right now?
Lead: Mostly speed. Our current solution takes 24 hours to sync. We need real-time.
Sales Rep: Our solution does sub-second syncing. However, it is pricier than what you use.
Lead: Price is a concern, but if it works, we have budget. My bigger worry is security compliance.
Sales Rep: We are SOC2 Type II compliant.
Lead: Okay, that removes that objection. Send me the contract.
"""

# Manually processing the mock data to fill the store
mock_docs = [Document(content=mock_transcript, meta={"source": "sales_call_001.mp3"})]

# We run just the embedding part of the pipeline manually for this demo
# (In production, the pipeline handles this automatically)

embedded_docs = doc_embedder.run(documents=mock_docs)["documents"]
document_store.write_documents(embedded_docs)

print("Indexing Complete.")

# 3. Build the RAG Pipeline (Google Embedder -> Retriever -> Google Gemini)
rag_pipeline = Pipeline()

# Google Embedder for the Query
text_embedder = GoogleGenAITextEmbedder(
    model=config.GEMINI_EMBEDDING_MODEL_NAME,
    api_key=Secret.from_token(config.GEMINI_API_KEY)
)

retriever = InMemoryEmbeddingRetriever(document_store=document_store)

# ChatPromptBuilder creates chat messages directly
template = [
    ChatMessage.from_user(
        """
You are a Sales Operations Analyst. 
Answer the question based ONLY on the following sales call transcripts.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ question }}
Answer:
"""
    )
]

prompt_builder = ChatPromptBuilder(template=template)

# Google Gemini Generator
# using 'gemini-1.5-flash' for speed and cost efficiency
gemini = GoogleGenAIChatGenerator(
    model=config.GEMINI_MODEL_NAME,
    api_key=Secret.from_token(config.GEMINI_API_KEY)
)

rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("gemini", gemini)

rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "gemini.messages")

# 4. Run Query
question = "What were the lead's main objections?"
print(f"\nQuestion: {question}")

result = rag_pipeline.run({
    "text_embedder": {"text": question},
    "prompt_builder": {"question": question}
})

answer = result["gemini"]["replies"][0].text
print("Answer:", answer)