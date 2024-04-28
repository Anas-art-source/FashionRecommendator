import chromadb
import openai
import pandas as pd

product_data = pd.read_csv('/home/khudi/Desktop/Farm/asli_farm.csv')

openai.api_key = ''
client = chromadb.PersistentClient(path="/home/khudi/Desktop/Farm/search/")
# collection = client.create_collection("my_information")


def text_embedding(text):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return response["data"][0]["embedding"]


collection = client.get_or_create_collection(name="farm_openai_2")

# print(collection)
def fetch_product(query):
    query_embedding = text_embedding(query)
    result = collection.query(query_embeddings=[query_embedding], n_results=5, include=["metadatas"])
    ids = result['ids'][0]
    result_return = []
    for id in ids:
        product = product_data[product_data['product_id'] == int(id)]
        result_return.append({
            "name": product['name'].iloc[0],
            "image_url": product['image_url'].iloc[0],
            "url": product['url'].iloc[0]
        })

    print(result_return)
    return result_return
