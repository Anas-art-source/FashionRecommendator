import pandas as pd


df = pd.read_csv('/home/khudi/Desktop/Farm/farm_dress_dataset.csv')

# df.head()
# print(df.shape[0])


# Remove duplicate rows
data_no_duplicates = df.drop_duplicates(subset=['product_id'])

# Display the DataFrame without duplicates

data_no_duplicates.to_csv('asli_farm.csv')