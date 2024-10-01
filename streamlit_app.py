import streamlit as st
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
import json
import warnings
from fake_useragent import UserAgent
import os
warnings.filterwarnings("ignore")

ua = UserAgent()
os.environ['USER_AGENT'] = ua.random

st.title("Job Helper")

llm = ChatGroq(
    groq_api_key='<api-key>',
    model_name='llama-3.1-70b-versatile'
)


def parse_resume(data):
    prompt = """
    Given a resume of candidate, extract all info from it in json format. {data}. Only return JSON data and nothing else NO PREAMBLE 
    """
    extract = PromptTemplate.from_template(prompt)
    chain_extract = extract | llm
    result = chain_extract.invoke(input={'data': data})

    try:
        return result.content
    except json.JSONDecodeError:
        return {"error": "Failed to parse resume data."}


def parse_job(content):
    extract = PromptTemplate.from_template(
        """`
        Given an HTML from a job website, you are required to return a json consisting of the following keys: ROLE, EXPERIENCE, SKILLS, DESCRIPTION. Only return a valid JSON (NO PREAMBLE)
        {content}
        Remember to return only JSON and not any other word
        """
    )
    chain_extract = extract | llm
    result = chain_extract.invoke(input={'content': content})
    return result.content


def scrape_page(url):
    loader = WebBaseLoader(url)
    try:
        content = loader.load().pop().page_content
        json_content = parse_job(content)
        return json_content
    except Exception as e:
        return {"error": f"Error scraping the URL: {str(e)}"}


def process_resume_and_job(resume, job):
    extract = PromptTemplate.from_template(
        """`
        Given a Resume JSON and a Job JSON. Return a brief feedback if the resume is in accordance to the job. 
        Also rate the resume. Only provide FEEDBACK and RATINGS and ways to improve
        RESUME: {resume}
        JOB: {job}
        NO PREAMBLE
        """
    )
    chain_extract = extract | llm
    result = chain_extract.invoke(input={'resume': resume, 'job': job})
    return result.content


st.header("Upload Resume and Job URL")
uploaded_resume = st.file_uploader("Choose a PDF file", type="pdf")
url_input = st.text_input(label="Enter Job URL")

if st.button("Submit"):
    resume_json = {}
    job_json = {}

    if uploaded_resume is not None:
        with st.spinner("Parsing resume..."):
            try:
                pdf_reader = PdfReader(uploaded_resume)
                data = "\n".join(page.extract_text() for page in pdf_reader.pages)
                resume_json = parse_resume(data)
            except Exception as e:
                st.error(f"Error reading PDF: {str(e)}")

    if url_input:
        if not url_input.startswith("http"):
            st.error("Please enter a valid URL.")
        else:
            with st.spinner("Scraping job information..."):
                job_json = scrape_page(url_input)

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Resume"):
            st.code(resume_json)

    with col2:
        with st.expander("Job"):
            st.code(job_json)

    if resume_json and job_json:
        feedback = process_resume_and_job(resume_json, job_json)
        st.markdown(feedback)