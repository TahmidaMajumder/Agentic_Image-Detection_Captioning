from tempfile import NamedTemporaryFile
import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
import os
import uuid

from tools import ImageCaptionTool, ObjectDetectionTool

# initialize agent tools
tools = [ImageCaptionTool(), ObjectDetectionTool()]

# memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True
)

# info about llm used

llm = ChatOpenAI(
     openai_api_key='YOUR_API_KEY',
     temperature=0,
     model_name="gpt-3.5-turbo"
 )



# agent params
agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method="generate"
)

# set title
st.title('Ask a question for the uploaded image')

# set header
st.header('Please upload an image')

# upload file
file = st.file_uploader("", type=['jpeg', 'jpg', 'png'])

if file:
    st.image(file, use_container_width=True)
    user_question = st.text_input('Ask a question about your image: ')

    # Generate a unique filename
    filename = f"temp_{uuid.uuid4().hex}.jpg"
    image_path = os.path.join("temp_images", filename)

    # Ensure temp directory exists
    os.makedirs("temp_images", exist_ok=True)

    # Write image to disk
    with open(image_path, "wb") as f:
        f.write(file.getbuffer())

    if user_question:
        with st.spinner("In progress..."):
            try:
                response = agent.run(f"{user_question}, this is the image path: {image_path}")
                st.write(response)
            finally:
                # Clean up only after processing is done
                os.remove(image_path)