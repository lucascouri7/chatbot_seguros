from llama_index.readers import SimpleDirectoryReader
from llama_index.node_splitter import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb


class IndexingPipeline:
    """Classe responsável por todo o processo de carregamento de documentos, 
    divisão em nós, configuração de armazenamento e criação de índices."""

    def __init__(self, input_dir: str, db_path: str, collection_name: str, embed_model: HuggingFaceEmbedding):
        self.input_dir = input_dir
        self.db_path = db_path
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.vector_store = None
        self.storage_context = None

    def run_pipeline(self, chunk_size: int = 512, batch_size: int = 50):
        """Executa toda a pipeline de indexação."""
        # Carregar documentos
        documents = SimpleDirectoryReader(input_dir="./data/").load_data()

        # Dividir documentos em nós
        nodes = self._split_documents_into_nodes(documents, chunk_size)

        # Inicializar o vector store
        self._initialize_storage_context()   

        # Criar e armazenar índices
        self._create_and_store_indices(nodes, batch_size)


    def _split_documents_into_nodes(self, documents, chunk_size):
        """Divide os documentos em nós, com base no tamanho do chunk especificado."""
        splitter = SentenceSplitter(chunk_size=chunk_size)
        return splitter.get_nodes_from_documents(documents)

    def _initialize_storage_context(self):
        """Inicializa o contexto de armazenamento usando Chroma."""
        db = chromadb.PersistentClient(path=self.db_path)
        chroma_collection = db.get_or_create_collection(self.collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def _create_and_store_indices(self, nodes, batch_size):
        """Cria e armazena índices em lotes, usando o modelo de embeddings."""
        for batch in self._iter_batch(nodes, batch_size):
            VectorStoreIndex(
                batch,
                storage_context=self.storage_context,
                embed_model=self.embed_model
            )

    def _iter_batch(self, nodes, batch_size):
        """Gera lotes de nós para indexação."""
        for i in range(0, len(nodes), batch_size):
            yield nodes[i:i + batch_size]


if __name__ == "__main__":
    # Carregar o modelo de embeddings fora da classe
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

    # Inicializar e rodar a pipeline de indexação
    pipeline = IndexingPipeline(
        input_dir="./data/",
        db_path="./chroma_db_bgeen",
        collection_name="seguros_collection_bgeen",
        embed_model=embed_model
    )
    pipeline.run_pipeline(chunk_size=512, batch_size=50)