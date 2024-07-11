# api = "AIzaSyDoR5Qf0_LjHKzn-cdThi42DLpn7s47BJc"
import pathlib
import textwrap
import streamlit as st
import google.generativeai as genai

import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# The below command is used to see the list of gemini models
# for m in genai.list_models():
#   if 'generateContent' in m.supported_generation_methods:
#     print(m.name)

# We are using gemini-1.5-flash for text ony prompts
model = genai.GenerativeModel('gemini-1.5-flash')
def get_gemini_response(question):
    response = model.generate_content(question)
    return response.text
# Let's create our streamlit application

st.set_page_config(page_title="Q&A Demo")
st.header("Gemini LLM Application")
input=st.text_input("Input: ",key="input")
submit = st.button("Ask the Question")

# When Submit is clicked

if submit:
    response = get_gemini_response(input)
    st.subheader("The Response is")
    st.write(response)