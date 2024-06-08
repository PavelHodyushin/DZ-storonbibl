import pandas as pd
import requests
import pprint
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


 # 1 библиотека Requests - простая и элегантная HTTP-библиотека для отправки запросов HTTP/1.1.

response = requests.get('https://api.github.com') # Запрос данных с помощью библиотеки requests из API
response_json = response.json()
pprint.pprint(response_json) #  Вывод их в консоль.
print(type(response_json))

 # 2 библиотека pandas - пакет на Python для анализа данных с гибкими и выразительными структурами.

df = pd.read_csv('titanic_new.csv', sep=';')
titanic = pd.DataFrame(df)
print(titanic) # Вывод ДатаФрейм
print(df.dtypes) # Проверяем тип данных в таблице
print(titanic.head(5)) # Вывод первых пяти строк
print(titanic[['age']]) # Вывод столбца возраста
if not titanic.empty:   # Если таблица не пустая
    mean_value = titanic['age'].mean()  # Вычисление среднего значения столбца 'value'
    print("Среднее значение столбца 'age':", mean_value)
print(titanic[["Sex", "age"]].groupby("Sex").mean()) # Вывод среднего значения возраста по каждому полу
print(titanic.groupby(["Sex", "Pclass"])["age"].mean()) # Вывод среднего возраста для каждой из комбинаций пола и класса


  # 3 библиотека Использование библиотеки numpy

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(np.linspace(0, 5, 10))
print(arr)   # Вывод массива
print(arr.size)     # Вывод количества элементов массива
print(arr.dtype)     # атрибут dtype показывает тип данных отдельного элемента
print(np.sum(arr))    # Сумма элементов массива
print(np.square(arr))     # возведение каждого элемента массива в квадрат
arr_2D = np.array([[1, 2, 3], [4, 5, 6]]) # Двумерный массив (матрица)
print(arr_2D)
print(arr_2D.size)   # Вывод количества элементов массива
print(arr_2D.ndim)    # кол_во строк
print(arr.ndim)    # чтобы узнать сколько у массива измерений

a = np.array([1, 2, 3, 4]) # Массивы из NumPy поддерживают все стандартные арифметические операции
print(a)
print(a + 3)
print(a - 2)
print(a * 2)
print(a / 2)
print(a ** 2)

  # библиотека 4 matplotlib

plt.xlabel('class')
plt.ylabel('age')
plt.scatter(x=titanic['Pclass'], y=titanic['age'])   # Диаграмма рассеяния (Scatter plot) в matplotlib
plt.show()  # Вывод диаграммы

df = pd.read_csv('titanic_new.csv', sep=';')

df.plot(x="PassengerId", y="age")    # линейный график с возростом
plt.show()

df.plot(x="PassengerId", y="age", kind='hist')    # вывод гистограммы возраста
plt.show()

df.plot(x="PassengerId", y="Pclass", kind='pie')    # вывод круговой диаграммы
plt.show()


  # 5 библиотека pillow

# Открываем изображение
img = Image.open('KINO.jpg')

# Изменяем размер изображения
resized_img = img.resize((512, 512))

# Сохраняем обработанное изображение
resized_img.save('resized_image.jpg')

print("Изображение обработано и сохранено.")