# AIzaSyAVjGeK4ZISXW-NrWgoHif1Oq0o1S1Kw8o

import google.generativeai as genai
import PIL.Image
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import time 
import os

data = pd.read_csv('/home/khudi/Desktop/Farm/farm_dress_dataset.csv')


# URL of the image
url = 'https://farmrio.com/cdn/shop/files/321137_01_e8778634-fd57-47d5-b6a8-6fa2d8fd99cf.jpg?v=1713475181&width=400'

genai.configure(api_key='AIzaSyAVjGeK4ZISXW-NrWgoHif1Oq0o1S1Kw8o')


prompt = """
Analyze the provided image of a dress and generate a JSON object containing detailed descriptions of its features. The JSON object should include the following fields:

1. `Silhouette`: Describe the overall shape of the dress.
2. `Fit`: Specify how tight or loose the dress appears.
3.`Front Neckline and Straps Shape`: Detail the shape of the neckline and the style of any straps.
4. `Front Neckline and Straps Depth`: Describe the depth of the neckline (e.g., deep, shallow).
5. `Front Neckline and Straps Detailing Style`: Note any special details like lace, ruffles, or embellishments on the neckline or straps.
6. `Sleeve Fit and Style`: Describe the fit (tight, loose) and style (e.g., puff, bell) of the sleeves.
7. `Sleeve Length`: Mention the length of the sleeves (e.g., short, long, three-quarter).
8. `Sleeve Detailing Style`: Describe any specific details on the sleeves, like embroidery or sequins.
9. `Waist Style`: Describe the style of the waist (e.g., fitted, empire, drop-waist).
10. `Length`: Note the overall length of the dress (e.g., mini, midi, maxi).
11. `Hem Style`: Describe the style of the hem (e.g., straight, ruffled, asymmetric).
12. `Pattern`: Identify any patterns present on the dress (e.g., floral, stripes).
13. `Occasions`: Suggest specific occasions where the dress could be worn, focusing on its suitability for various settings and activities.
14. `Persona`: Describe the likely buyer persona for this dress, focusing on their age, fashion aesthetic, and lifestyle preferences.
15. Predominant Colors: List the main colors visible in the dress.
16. Background Color: Describe the background color in the image, if applicable.
## Example Output:
{
  "Silhouette": "A-line",
  "Fit": "Loose",
  "Front Neckline and Straps Shape": "V-neck with thin straps",
  "Front Neckline and Straps Depth": "Medium",
  "Front Neckline and Straps Detailing Style": "Lace trimming",
  "Sleeve Fit and Style": "Sleeveless",
  "Sleeve Length": "N/A",
  "Sleeve Detailing Style": "N/A",
  "Waist Style": "Cinched at the natural waist",
  "Length": "Midi",
  "Hem Style": "Straight",
  "Pattern": "Floral print",
  "Occasions": "This floral midi dress is ideal for events such as garden parties, casual weddings, or daytime outings. It pairs well with light, elegant accessories and comfortable footwear for a stylish yet effortless look.",
  "Persona": "The buyer persona for this dress is a woman in her late 20s to early 40s who values elegance and simplicity. She prefers versatile, feminine pieces that can transition smoothly from casual to semi-formal occasions."
"Predominant Colors": ["blue", "green", "pink"],
  "Background Color": "white"
  }


"""

def get_response(image):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([prompt, image], stream=True)
    response.resolve()
    return response.text


import json

def append_to_jsonl(file_path, data):
    """
    Appends a new JSON object to a JSON Lines file and saves it.
    
    :param file_path: Path to the JSONL file.
    :param data: Dictionary containing the data to append.
    """
    # Ensure that data is a dictionary
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary.")
    
    # Open the file in append mode and write the new data
    with open(file_path, 'a') as file:
        # Convert the dictionary to a JSON string and write it to the file
        json_string = json.dumps(data)
        file.write(json_string + '\n')  # Append a newline character to keep JSON Lines format

past_product_id = None

for i, row in data.iterrows():
    if row['product_id'] == past_product_id:
        continue
    past_product_id = row['product_id']
    try:
        time.sleep(3)
        # Send an HTTP GET request to the URL
        response = requests.get(row['image_url'])

        # Check if the request was successful
        if response.status_code == 200:
            # Open the image from the bytes in response content
            img = Image.open(BytesIO(response.content))
            img.show()  # Display the image using the default image viewer
        else:
            print("Failed to retrieve the image. Status code:", response.status_code)

        response_json = get_response(img)
        res_json = json.loads(response_json)
        res_json['product_id'] = row['product_id']
        append_to_jsonl('farm_feature_list.jsonl',res_json)

    except Exception as e:
        continue

