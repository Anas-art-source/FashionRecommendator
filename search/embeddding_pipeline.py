from typing import List
from together import Together
from langchain_together.embeddings import TogetherEmbeddings
import pandas as pd
import chromadb
import os
from tqdm import tqdm
import openai

openai.api_key = ''

os.environ["OPENAI_API_KEY"] = ""

os.environ['TOGETHER_API_KEY'] = ''
client = chromadb.PersistentClient(path="/home/khudi/Desktop/Farm/search/")

# client = Together(api_key='84ee7f7050742d032694d0f70e9b628e4293246d739df04b02721ed68502fb76')
collection = client.create_collection(name="farm_openai_3")


embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

data= pd.read_csv('/home/khudi/Desktop/Farm/farm_feature_list.csv')

print(data.head())
print(data.columns)


def text_embedding(text):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return response["data"][0]["embedding"]

for i, row in tqdm(data.iterrows(), total=len(data)):
    embedding_template = f"""
    Silhouette: {row['Silhouette']}
    Fit: {row['Fit']}
    Front Neckline and Straps Shape: {row['Front Neckline and Straps Shape']}
    Front Neckline and Straps Depth: {row['Front Neckline and Straps Depth']}
    Front Neckline and Straps Detailing Style: {row['Front Neckline and Straps Detailing Style']}
    Sleeve Fit and Style: {row['Sleeve Fit and Style']}
    Sleeve Detailing Style: {row['Sleeve Detailing Style']}
    Waist Style: {row['Waist Style']}
    Length: {row['Length']}
    Hem Style: {row['Hem Style']}
    Occasions: {row['Occasions']}
    Persona: {row['Persona']}
    Predominant Colors: {row['Predominant Colors']}
    """

    embedding = text_embedding(embedding_template)
    collection.add(
    embeddings=embedding,
    documents=embedding_template,
    ids=str(row['product_id'])
    # metadatas=[{'M': 1, "S": 1}]

        )


# results = collection.query(

#     query_texts=["This is a query document"],

#     n_results=2,

#     where={"M": {"$e":1}, "S": {"$e":1}, "L": {"$e":1}}

# )

# print(embeddings)/
# [[0.13437459, 0.09866201, ..., -0.20736569]]