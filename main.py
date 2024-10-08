from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.storage.chat_store import SimpleChatStore
from contextlib import asynccontextmanager
import startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega os modelos de embeddings e llm
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
    llm = Ollama(model="llama3.2:3b", request_timeout=420.0)

    # Inicializa o índice do vector store
    index_manager = startup.IndexManager(db_path="./chroma_db_bgeen",
                                         collection_name="seguros_collection_bgeen",
                                         embed_model=embed_model)
    index = index_manager.load_index()

    # Inicializa o chat engine a partir do índice criado
    chat_manager = startup.ChatEngineManager(index, llm, 3000, 120)
    app.state.chat_store = chat_manager.initialize_chat_store()
    prompt = ("Você é um chatbot com objetivo de fornecer informações e tirar dúvidas sobre seguros de automóveis de 4 empresas: "
                "Santander, SUHAI, Bradesco e Porto Seguro."
                "Aqui estão os documentos relevantes para o contexto:\n"
                "{context_str}\n"
                "Instrução: Utilize o contexto acima para responder e ajudar o usuário. Não diga explicitamente que está respondendo baseado em um documento.\n"
                "Pergunta: {query_str}\n"
                "Resposta: ")
    app.state.chat_engine = chat_manager.initialize_chat_engine(prompt, 2, 0.75)
    app.state.chat_store = SimpleChatStore()
    yield


# Cria a instância do FastAPI com o lifespan
app = FastAPI(lifespan=lifespan)


# Modelo para o input JSON do usuário
class MessageInput(BaseModel):
    message: str = Field(..., min_length=1, description="Message field must be a non-empty string")


# Endpoint para o chatbot
@app.post("/chat/")
async def chatbot(message_input: MessageInput):
    user_input = message_input.message
    if not user_input:
        raise HTTPException(status_code=400, detail="É necessário escrever uma mensagem!")
    response = process_user_input(user_input)
    log_interaction()
    return {"response": response}


# Função para processar a entrada do usuário e consultar o modelo
def process_user_input(user_input):
    chat_engine = app.state.chat_engine
    response = chat_engine.chat(user_input)
    return response


# Função para armazenar logs (pode conectar a um banco de dados)
def log_interaction():
    chat_store = app.state.chat_store
    chat_store.persist(persist_path="chat_store.json")
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
