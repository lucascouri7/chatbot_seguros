from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
import chromadb

class IndexManager:
    """Classe responsável por carregar o index do VectorStore."""
    
    def __init__(self, db_path: str, collection_name: str, embed_model: HuggingFaceEmbedding):
        """
        Inicializa a classe IndexManager com os parâmetros fornecidos.
        
        :param db_path: Caminho do banco de dados Chroma persistente.
        :param collection_name: Nome da coleção do ChromaDB.
        :param embed_model: Nome do modelo de embeddings a ser utilizado.
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.index = None

    def load_index(self):
        """
        Inicializa o index do vector store.
        
        :return: Objeto do índice VectorStoreIndex.
        """
        chroma_collection = self._get_chroma_collection()

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model,
        )
        return self.index

    def _get_chroma_collection(self):
        """
        Inicializa e retorna a coleção do ChromaDB.

        :return: Objeto da coleção Chroma.
        """
        db_client = chromadb.PersistentClient(path=self.db_path)
        return db_client.get_or_create_collection(self.collection_name)



class ChatEngineManager:
    """Classe responsável por gerenciar o chat engine e o armazenamento de conversas."""
    
    def __init__(self, index: VectorStoreIndex, llm_model: Ollama, token_limit: int = 3000, request_timeout: float = 420.0):
        """
        Inicializa a classe ChatEngineManager com os parâmetros fornecidos.
        
        :param index: Objeto do índice VectorStoreIndex que será utilizado pelo chat engine.
        :param llm_model: Nome do modelo de linguagem a ser utilizado pelo chat engine.
        :param token_limit: Limite de tokens para o chat memory buffer.
        :param request_timeout: Tempo limite para requisições ao modelo de linguagem.
        """
        self.index = index
        self.llm_model = llm_model
        self.token_limit = token_limit
        self.request_timeout = request_timeout
        self.chat_store = None
        self.chat_memory = None
        self.chat_engine = None
    
    def initialize_chat_store(self):
        """
        Inicializa o armazenamento de conversas (chat store) e o buffer de memória do chat.
        
        :return: Objeto do chat_store e chat_memory.
        """
        self.chat_store = SimpleChatStore()
        self.chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=self.token_limit,
            chat_store=self.chat_store,
            chat_store_key="user1",
        )
        return self.chat_store
    
    def initialize_chat_engine(self, context_prompt: str, similarity_top_k: int = 2, temperature: float = 0.75, verbose: bool = False):
        """
        Inicializa o motor de chat (chat engine) com o índice e as configurações fornecidas.
        
        :param context_prompt: Instrução de contexto usada pelo chat engine para responder perguntas.
        :param similarity_top_k: Quantidade de resultados similares que o motor de chat deve buscar.
        :param temperature: Grau de criatividade nas respostas.
        :param verbose: Se o motor deve ser verboso.
        :return: Objeto do chat engine.
        """

        # Configura o chat engine
        self.chat_engine = self.index.as_chat_engine(
            chat_mode="context",
            similarity_top_k=similarity_top_k,
            temperature=temperature,
            memory=self.chat_memory,
            context_prompt=context_prompt,
            llm=self.llm_model,
            verbose=verbose
        )
        return self.chat_engine
