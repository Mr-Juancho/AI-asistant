# memory.py
import os
import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
# --- CONFIGURACIÓN ---
KNOWLEDGE_BASE_DIR = "./myKnowledgeBase"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"

class MemoryManager:
    """Gestiona la creación y consulta de la base de conocimiento."""
    def __init__(self):
        """Inicializa los componentes de la base de conocimiento."""
        print("🧠 Inicializando la memoria del asistente...")
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
        # Conecta a la base de datos persistente de ChromaDB
        self.vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self.embeddings
        )
        
        # Crea un 'retriever' para buscar en la base de datos
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        print("✅ Memoria lista.")

    # En memory.py, dentro de la clase MemoryManager

    async def get_context(self, query: str) -> str:
        """Busca contexto relevante en la base de conocimiento."""
        print(f"🔍 Buscando en la memoria sobre: '{query}'")

        # --- LÍNEA A CAMBIAR ---
        # Antes: relevant_docs = await self.retriever.aget_relevant_documents(query)
        # Después:
        relevant_docs = await self.retriever.ainvoke(query)

        if not relevant_docs:
            return "No se encontró información relevante en la base de conocimiento."

        context = "\n---\n".join([doc.page_content for doc in relevant_docs])
        print("✅ Contexto encontrado.")
        return context

# En memory.py

def build_memory():
    """Lee documentos, los divide, crea embeddings y los guarda en ChromaDB."""
    print("Construyendo la base de conocimiento desde los archivos...")
    
    # --- LÍNEA CORREGIDA ---
    # Ahora busca archivos .txt, .md y .pdf y usa el cargador apropiado para cada uno
    # de forma automática, al no especificar 'loader_cls'.
    loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*[.txt,.md,.pdf]", show_progress=True)
    documents = loader.load()
    
    if not documents:
        print(f"No se encontraron documentos en la carpeta '{KNOWLEDGE_BASE_DIR}'. La memoria no fue construida.")
        return

    # Divide los documentos en trozos más pequeños (chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Crea la base de datos de vectores con los textos y la guarda en disco
    print(f"Creando y guardando {len(texts)} fragmentos de texto en la base de datos...")
    Chroma.from_documents(
        texts,
        OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DB_DIR
    )
    print("¡Base de conocimiento construida y guardada exitosamente!")

# --- Bloque para ejecutar la construcción de la memoria ---
if __name__ == "__main__":
    build_memory()