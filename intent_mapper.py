import os
# from together import Together
from jinja2 import Template
from langchain_together import Together
import pandas as pd
import re
import json 
from search.query_data import fetch_product
import streamlit as st

os.environ['TOGETHER_API_KEY'] = '84ee7f7050742d032694d0f70e9b628e4293246d739df04b02721ed68502fb76'


products = pd.read_csv('/home/khudi/Desktop/Farm/farm_feature_list.csv')

# print(products.head())
# client = Together()


system_prompt = """
You are an expert recommendation system. Your task is to help business teams to analyze the most important feature the user is interested in (not just raw summarization). 
Given the user's browsing history of fashion items, analyze the common features and preferences exhibited across the items. Consider attributes such as color, pattern, length, sleeve length, neckline, and material. Based on these insights, create a JSON object that captures the most relevant user's interest map, highlighting preferred styles, colors, patterns, and other specific attributes. The JSON should be structured to clearly delineate each category of interest for potential use in personalized recommendations or targeted marketing. 

## Output Format:
Your output should be in `json` with following keys and values:
* `Analysis_of_Primary_Preffered_Style` - detail analysis of user primary preferred style in natural language
* `Analysis_of_Primary_Preffered_Color` - detail analysis of user primary preferred color in natural language
* `Analysis_of_Primary_Preffered_Pattern` - detail analysis of user primary preferred pattern in natural language
* `Analysis_of_Primary_Preffered_length` - detail analysis of user primary preferred length in natural language
* `Analysis_of_Primary_Preffered_Neckline_and_Straps` - detail analysis of user primary preference about neckline and strap, including shape, depth and detailing style in natural language
* `Analysis_of_Primary_Preffered_sleeve` - detail analysis of user primary preference about sleeve, including fit and style in natural language
* `Analysis_of_Primary_Preffered_waist_style` - detail analysis of user primary preference about waist style in natural language
* `Analysis_of_Primary_Preffered_fit` - detail analysis of user primary preference about fit in natural language
* `Analysis_of_user_persona` - detail analysis of user persona based on browsing history in natural language
* `Analysis_of_ocassion` - detail analysis of ocassion user is looking dress for in natural language

## Example Output
{
    "Analysis_of_Primary_Preffered_Style": "",
    "Analysis_of_Primary_Preffered_Color": "",
    "Analysis_of_Primary_Preffered_Pattern": "",
    "Analysis_of_Primary_Preffered_length": "",
    "Analysis_of_Primary_Preffered_Neckline_and_Straps": "",
    "Analysis_of_Primary_Preffered_sleeve": "",
    "Analysis_of_Primary_Preffered_waist_style": "",
    "Analysis_of_Primary_Preffered_fit": "",
    "Analysis_of_user_persona": "",
    "Analysis_of_ocassion": ""
}

Your output should strictly be only json with defined key and values. Wrapped JSON in <json>output in json format</json>


## User Detail
{{user_detail}}

 

Your output: 
"""

def get_user_intent(user_detail):
    template = Template(system_prompt)
    prompt = template.render(user_detail=user_detail)

    
    llm = Together(
        model="meta-llama/Llama-3-8b-chat-hf",
        temperature=0.5,
        max_tokens=600,
        top_k=1,
        together_api_key="84ee7f7050742d032694d0f70e9b628e4293246d739df04b02721ed68502fb76"
    )

    response = llm.invoke(prompt)

    # Extracting JSON content using regex
    json_content = re.search(r'<json>(.*?)</json>', response, re.DOTALL).group(1)

    # Parsing JSON
    parsed_json = json.loads(json_content)
    text = f"""
        The Preferred Style of user: {parsed_json['Analysis_of_Primary_Preffered_Style']}
        The Preferred Color of user: {parsed_json['Analysis_of_Primary_Preffered_Color']}
        The Preferred Pattern of user: {parsed_json['Analysis_of_Primary_Preffered_Color']}
        The Preferred Length of user: {parsed_json['Analysis_of_Primary_Preffered_length']}
        The Preferred Neckline and Straps of user:{parsed_json['Analysis_of_Primary_Preffered_Neckline_and_Straps']}
        The Preferred Sleeve of user: {parsed_json['Analysis_of_Primary_Preffered_sleeve']}
        The Preferred waist style: {parsed_json['Analysis_of_Primary_Preffered_waist_style']}
        The Preferred Fit of user: {parsed_json['Analysis_of_Primary_Preffered_fit']}
        The user persona: {parsed_json['Analysis_of_user_persona']}
        The ocassion user is looking for: {parsed_json['Analysis_of_ocassion']}

    """
    return text


def create_prompt(request):
    interactions = request['interactions']
    interaction_augmented_list = []
    for item in interactions:
        # print(type(item), item)
        product_id = int(item['product_id'])
        product = products[products['product_id'] == product_id]
        if product.shape[0] > 0:
            # print(product['Silhouette'].iloc[0])
            interaction_augmented_list.append({
                "timestamp": item['timestamp'],
                "About Product": {
                "Silhouette": product['Silhouette'].iloc[0],
                'Fit': product['Fit'].iloc[0],
                "Front Neckline and Straps Shape": product['Front Neckline and Straps Shape'].iloc[0],
                "Front Neckline and Straps Depth": product['Front Neckline and Straps Depth'].iloc[0],
                "Front Neckline and Straps Detailing Style": product['Front Neckline and Straps Detailing Style'].iloc[0],
                "Sleeve Fit and Style": product['Sleeve Fit and Style'].iloc[0],
                "Sleeve Length": product['Sleeve Length'].iloc[0],
                "Sleeve Detailing Style": product['Sleeve Detailing Style'].iloc[0],
                "Waist Style": product['Waist Style'].iloc[0],
                "Length": product['Length'].iloc[0],
                'Hem Style': product['Hem Style'].iloc[0],
                "Pattern": product['Pattern'].iloc[0],
                "Occasions": product['Occasions'].iloc[0],
                'Persona': product['Persona'].iloc[0],
                'Predominant Colors': product['Predominant Colors'].iloc[0]
                }
            })

    return interaction_augmented_list





interactions = pd.read_json('/home/khudi/Desktop/Farm/user_interactions.json')


def get_user_browsing_detail(user_id):
    return interactions[interactions['user_id'] == user_id].iloc[0]

# user_interaction = get_user_browsing_detail('user_5')
# user_detail = create_prompt(user_interaction)
# response = get_user_intent(user_detail=user_detail)
# fetch_product(response)/


prompt = st.chat_input("Say something")
if prompt:
    st.write(f"Get recommendation of user: {prompt}")

    user_interaction = get_user_browsing_detail(prompt)
    user_detail = create_prompt(user_interaction)
    response = get_user_intent(user_detail=user_detail)
    products = fetch_product(response)
    # Time of execution
    # Similarity score
    # Token
    st.markdown(products)

