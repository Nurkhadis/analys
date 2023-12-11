#%%
import pandas as pd
import numpy as np
import psycopg2
import seaborn as sns                       #visualization
import matplotlib.pyplot as plt             #visualization
import matplotlib_inline  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error




df = pd.read_csv("train.csv", low_memory=False)
#%%
print(df.head())

#%% Посмотрим на основную информацию о датасете, такую как типы данных и наличие пропущенных значений
print(df.info())

#%% Посмотрим на основные статистические характеристики числовых данных
print(df.describe())

#%% Проверим наличие пропущенных значений
print(df.isnull().sum())

#%% Удаление дубликатов (если они есть)
df = df.drop_duplicates()

# Обработка пропущенных значений (например, заполнение их средними значениями или удаление)
#%% Например, для заполнения пропущенных значений в числовых столбцах средними значениями:
df.fillna(df.mean(), inplace=True)

# Дополнительные шаги анализа и очистки могут включать в себя обработку категориальных данных, создание новых признаков и т. д.

#%% Сохранение очищенного датасета (если необходимо)
df.to_csv('cleaned_dataset.csv', index=False)

######################################################
##EDA
#%%
print(df)
# %% To display the top 5 rows 
df.head(5) 
# %% Чтобы отобразить нижние 5 строк
df.tail(5) 
# %% Проверка типов данных
df.dtypes
# %% Удаление ненужных столбцов
df = df.drop(['sc_h', 'sc_w'], axis=1)
df.head(5)
########################### VISUALZATION
# %% Визуализация распределения числовых признаков
df.hist(figsize=(12, 10))
plt.show()


# %% Исследование категориальных переменных
sns.countplot(x='n_cores', data=df)
plt.show()

# %% Сопоставьте различные объекты друг с другом (рассеяние), с частотой (гистограмма)
df.n_cores.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("n_cores by pc")
plt.ylabel('n_cores')
plt.xlabel('pc')
# %% Heat Maps
plt.figure(figsize=(10,5))
c= df.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c

# %% Scatterplot
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['battery_power'], df['clock_speed'])
ax.set_xlabel('battery_power')
ax.set_ylabel('clock_speed')
plt.show()

#####################################################
# %% Гипотеза: Существует положительная корреляция
#  между объемом оперативной памяти (RAM) в мобильном 
# устройстве и его объемом батареи. Другими словами, 
# по мере увеличения объема оперативной памяти, 
# объем батареи мобильного устройства также увеличивается.
print(df[['ram', 'battery_power']].describe())

# Визуализация отношения
plt.scatter(df['ram'], df['battery_power'])
plt.xlabel('RAM')
plt.ylabel('Battery')
plt.title('Связь между RAM и Battery')
plt.show()

# %% Анализ корреляции
correlation_ram_battery = df['ram'].corr(df['battery_power'])
print(f'Корреляция между RAM и Battery: {correlation_ram_battery}')

# %% Тестирование гипотезы
from scipy.stats import pearsonr

# Нулевая гипотеза: Нет корреляции
# Альтернативная гипотеза: Есть положительная корреляция
stat, p_value = pearsonr(df['ram'], df['battery_power'])

print(f'Коэффициент корреляции: {stat}\nЗначение p: {p_value}')

# Проверьте, если ли p-value ниже уровня значимости (например, 0,05)
if p_value < 0.05:
    print("Отклонить нулевую гипотезу")
else:
    print("Не удалось отклонить нулевую гипотезу")

# %% Построения регрессии
#################################################
# Загрузите ваш датасет
df = pd.read_csv('train.csv')

# Предполагаем, что 'ram', 'battery_power' и 'talk_time' - это признаки, которые вы хотите использовать
X = df[['ram', 'battery_power', 'talk_time']]  # Добавьте соответствующие признаки
y = df['price_range']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

plt.scatter(y_test, y_pred)
plt.xlabel('Фактические цены')
plt.ylabel('Предсказанные цены')
plt.title('Модель регрессии: фактические цены против предсказанных цен')
plt.show()




# %%
