import os
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings  # For embeddings from OpenAI models
from langchain.vectorstores import FAISS  # For FAISS vector store
from langchain.chat_models import ChatOpenAI  # For OpenAI's chat-based models (e.g., GPT-3.5, GPT-4)
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder  # For custom prompt templates
from langchain.memory import ConversationBufferMemory  # For conversational memory
from langchain.chains import ConversationalRetrievalChain  # For creating a conversational chain

import os
from dotenv import load_dotenv

load_dotenv('OpenAI.env')

# Set OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load FAISS vector store
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.load_local("my_faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Create a custom chat prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Medical Assistant Chatbot. Use the following context to answer the user's question. If you don't know the answer, say you don't know."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Set up conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Initialize the chat model (you can change the model here)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Create the conversational chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True,
)

# Start chat loop
print("Chatbot ready! Type 'exit' to quit.")
while True:
    # Prompt for user input
    user_input = input("You: ")

    # Exit condition
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chatbot.")
        break

    # Get the answer from the conversational chain
    result = chain({"question": user_input})

    # Display the answer
    print("Bot:", result["answer"])
