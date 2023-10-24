#%%
import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
import matplotlib_inline     
sns.set(color_codes=True)

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
#%%
#Загрузка данных в фрейм данных
print(df)
#%%

print(df.head(5))
# %%
df.tail(5)
# %%
#Проверка типов данных
df.dtypes
# %%
df = df.drop(['mpg', 'disp', 'wt', 'vs', 'am'], axis=1)
df.head(5)
# %%
#Переименование столбцов
df = df.rename(columns={"mpg": "MPGNEW", "disp": "Disperse", "wt": "WTTT", "vs": "VVSS", "am": "MA"})
df.head(5)
# %%
#Удаление повторяющихся строк
df.shape
# %%
duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)
# %%
df.count()
# %%
df = df.drop_duplicates()
df.head(5)
# %%
df.count()
# %%
df = df.dropna()    # Удаление пропущенных значений.
df.count()
# %%
#Удаление пропущенных или нулевых значений
print(df.isnull().sum())   # После удаления значений
# %%
#Обнаружение выбросов
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
# %%
df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape
# %%
#Сопоставление различных объектов друг с другом, с частотой
df.model.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("model")
plt.ylabel('cyl')
plt.xlabel('drat');
# %%
plt.figure(figsize=(10,5))
c= df.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c

# %%
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['HP'], df['Price'])
ax.set_xlabel('HP')
ax.set_ylabel('Price')
plt.show()
# %%
