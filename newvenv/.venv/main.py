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

#connection = sqlite3.connect("sample_database")

if connection:
    print("connection is set...")
else:
    print("connection is not set...")

# cusor = connection.cursor()
# query_car_list = """select "model", "mpg", "cyl", "disp" from "mtcars" """
# cusor.execute(query_car_list)
# car_list = cusor.fetchall()
# print("model", " | ", "mpg", " | ", "cyl", " | ", "disp")
# for row in car_list:
#     print(row[0], " | ", row[1], " | ", row[2], " | ", row[3])


query = "select * from mtcars"
df = pd.read_sql_query(query, connection)

print(df)