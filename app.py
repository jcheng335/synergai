import json
import os
import sys
import boto3
import streamlit as st

# We will be using Titan Embeddings Model to generate Embedding
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock

# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding and Vector Store
import faiss
from langchain_community.vectorstores import FAISS

# LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Set the region
region_name = "us-east-1"

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime", region_name=region_name)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Initialize AWS clients for transcription and connect
transcribe = boto3.client('transcribe', region_name=region_name)
connect = boto3.client('connect', region_name=region_name)
dynamodb = boto3.resource('dynamodb', region_name=region_name)
prompts_table = dynamodb.Table('InterviewPrompts')

# Data ingestion
def parse_resume(file):
    if file.name.endswith('.pdf'):
        loader = PyPDFDirectoryLoader(path=os.path.dirname(file.name))
        documents = loader.load()
        text = documents[0].content
    elif file.name.endswith('.docx'):
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError('Unsupported file format')

    splitter = RecursiveCharacterTextSplitter()
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(docs):
    vectors = [bedrock_embeddings.embed(doc) for doc in docs]
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors).astype('float32'))
    return index

def get_llama3_llm():
    # Create the Llama 3 Model
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len':512})
    return llm

# Prompt template to generate new questions
def create_prompt(context, standard_questions, transcript):
    return f"""
    Based on the following resume content, standard interview questions, and the interviewee's recent response, generate a new follow-up interview question tailored to the interviewee.

    Resume Content:
    {context}

    Standard Interview Questions:
    {standard_questions}

    Interviewee's Response:
    {transcript}

    New Follow-Up Question:
    """

def generate_question(llm, context, standard_questions, transcript):
    prompt = create_prompt(context, standard_questions, transcript)
    response = llm({"text": prompt})
    return response['choices'][0]['text']

def start_transcription_streaming(resume_text, standard_questions):
    import asyncio
    from transcribe_streaming import TranscribeStreamingClient  # Assuming you have the AWS Transcribe Streaming SDK
    
    async def transcribe_streaming():
        transcribe_client = TranscribeStreamingClient()
        stream = await transcribe_client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=16000,
            media_encoding="pcm"
        )
        
        async def write_chunks():
            with open("audio.wav", "rb") as f:
                while chunk := f.read(4096):
                    await stream.input_stream.send_audio_event(audio_chunk=chunk)
            await stream.input_stream.end_stream()
        
        async def read_chunks():
            async for event in stream.output_stream:
                if isinstance(event, TranscriptResultStream.TranscriptEvent):
                    transcript = event.transcript.results[0].alternatives[0].transcript
                    st.write("Interviewee Response: ", transcript)
                    llm = get_llama3_llm()
                    new_question = generate_question(llm, resume_text, standard_questions, transcript)
                    st.write("Generated Follow-Up Question: ", new_question)

        await asyncio.gather(write_chunks(), read_chunks())
    
    asyncio.run(transcribe_streaming())

def store_standard_questions(prompts):
    for prompt in prompts:
        prompts_table.put_item(Item={'prompt': prompt})

def get_standard_questions():
    response = prompts_table.scan()
    return [item['prompt'] for item in response['Items']]

def start_session(phone_number):
    instance_id = os.environ.get('CONNECT_INSTANCE_ID', 'your_instance_id')
    contact_flow_id = os.environ.get('CONNECT_CONTACT_FLOW_ID', 'your_contact_flow_id')
    response = connect.start_outbound_voice_contact(
        DestinationPhoneNumber=phone_number,
        ContactFlowId=contact_flow_id,
        InstanceId=instance_id,
        SourcePhoneNumber=os.environ.get('SOURCE_PHONE_NUMBER', 'your_source_phone_number')
    )
    return response

# Streamlit UI
st.title("Interview Assistant App")

# Upload resume
st.header("Upload Resume")
uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])

resume_vector = None
resume_text = None

if uploaded_file is not None:
    resume_chunks = parse_resume(uploaded_file)
    resume_vectors = [bedrock_embeddings.embed(chunk) for chunk in resume_chunks]
    resume_vector = np.mean(resume_vectors, axis=0)
    faiss_index = faiss.IndexFlatL2(resume_vector.shape[0])
    faiss_index.add(np.array(resume_vectors).astype('float32'))
    resume_text = "\n".join(resume_chunks)
    st.success("Resume parsed and stored successfully.")

# Import standard prompts
st.header("Import Standard Prompts")
prompts = st.text_area("Enter standard prompts (one per line)")
if st.button("Import Prompts"):
    prompts_list = prompts.split("\n")
    store_standard_questions(prompts_list)
    st.success("Prompts imported successfully.")

# Get standard prompts
st.header("Get Standard Prompts")
if st.button("Get Prompts"):
    standard_questions = get_standard_questions()
    st.write(standard_questions)

# Generate interview questions
st.header("Generate Interview Questions")
if st.button("Generate Questions"):
    if uploaded_file and standard_questions:
        llm = get_llama3_llm()
        initial_question = generate_question(llm, resume_text, standard_questions, "")
        st.write("Initial Question: ", initial_question)
    else:
        st.warning("Please upload a resume and import standard interview questions first.")

# Real-time transcription
st.header("Real-time Transcription")
if st.button("Start Real-time Transcription"):
    if resume_vector and standard_questions:
        start_transcription_streaming(resume_text, standard_questions)
        st.success("Real-time transcription started.")
    else:
        st.warning("Please upload a resume and import standard interview questions first.")

# Start communication session
st.header("Start Communication Session")
phone_number = st.text_input("Phone Number")
if st.button("Start Session"):
    start_session(phone_number)
    st.success("Session started.")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        st.run()
    else:
        st.warning("Run the script with 'python app.py run' to start the Streamlit server.")
