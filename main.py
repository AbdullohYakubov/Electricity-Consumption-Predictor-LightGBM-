# import csv
# import matplotlib.pyplot as plt
# import numpy as np

# try:
#     with open ('01.01.2022-15.04.2022(reading).csv', newline='') as csvfile:
#         csv_reader = csv.reader(csvfile)
#         list = [[element for element in row] for row in csv_reader]

# except FileNotFoundError:
#     print('dart_hits.csv file not found!')

# else:
#     years = [list[i][1] for i in range(1, len(list))]
#     reading = [float(list[i][2]) for i in range(1, len(list))]

#     # print(reading)

# try:

#     fig, ax = plt.subplots()

#     ax.scatter(years, reading, color='black', marker='+')

#     ax.set_title("Scatterplot", pad=15, color='red')
#     ax.set_xlabel('Years', loc='right')
#     ax.set_ylabel('Reading', loc='top', rotation='horizontal')
#     ax.axis([-len(years), max(reading), len(years), min(reading)])
#     ax.tick_params(colors='blue')
#     ax.xaxis.label.set_color('blue')
#     ax.yaxis.label.set_color('blue')
#     ax.spines['left'].set_color('black')
#     ax.spines['left'].set_position('zero')
#     ax.spines['bottom'].set_color('black')
#     ax.spines['bottom'].set_position('zero')
#     ax.spines['top'].set_color('none')
#     ax.spines['right'].set_color('none')

#     plt.show()

# except:
#     print('Variable years or reading is not defined!')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. Загрузка данных
df = pd.read_csv('01.01.2022-15.04.2022(reading).csv')

# Преобразуем столбец с датой в datetime
df['reading_date'] = pd.to_datetime(df['reading_date'])

# Преобразуем дату в число (например, количество дней с первой даты)
df['days'] = (df['reading_date'] - df['reading_date'].min()).dt.days

X = df[['days']].values  # используем оригинальные значения
y = df['reading'].values

# 2. Визуализация
plt.figure(figsize=(18, 6))
plt.scatter(X.flatten(), y, s=8, alpha=0.7)
plt.xlabel('X', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('Scatter plot', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0, df['days'].max())
plt.ylim(0, 60000000)  # yoki plt.ylim(0, df['reading'].max())
plt.show()

# 3. Линейная регрессия
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_linear = lin_reg.predict(X)
mse_linear = mean_squared_error(y, y_pred_linear)

# 4. Полиномиальная регрессия (например, 2-й степени)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)
mse_poly = mean_squared_error(y, y_pred_poly)

print(f"MSE линейной модели: {mse_linear}")
print(f"MSE полиномиальной модели: {mse_poly}")

# 5. Визуализация моделей
plt.scatter(X, y, color='blue', s=8, alpha=0.7)
plt.plot(X, y_pred_linear, color='red', label='Линейная')
plt.plot(X, y_pred_poly, color='green', label='Полиномиальная')
plt.legend()
plt.show()

print("Максимальное количество дней:", df['days'].max())
print("Максимальное значение reading:", df['reading'].max())