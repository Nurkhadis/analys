#%%
import sqlite3
import psycopg2
import pandas as pd
import numpy as np

connection = psycopg2.connect(
    database="dataanalystest",
    user="postgres",
    password="1176",
    host="localhost",
    port=5432,
)


if connection:
    print("connection is set...")
else:
    print("connection is not set...")


query = "select * from mtcars"
df = pd.read_sql_query(query, connection)

print(df)
# #%%
# print(df.head(5))
# # %%
# df.tail(5)
# # %%
# df.dtypes
# # %%
# df = df.drop(['mpg', 'disp', 'wt', 'vs', 'am'], axis=1)
# df.head(5)
# # %%
# df = df.drop(['mpg', 'disp', 'wt', 'vs', 'am'], axis=1)
# df.head(5)