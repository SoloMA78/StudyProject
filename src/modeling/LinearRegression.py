# Class to implement LinearRegression

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from pathlib import Path


class LRPredict:
    range_limits = {
        'IW': [43, 49],
        'IF': [131, 150],
        'VW': [4.5, 12],
        'FP': [50, 125]
    }

    def __init__(self):
        if Path('lr_model').exists():
            f = open('lr_model', 'rb')
        self.pipe = pickle.load(f)

    @staticmethod
    def fit(data_path):
        try:
            df = pd.read_csv(data_path)
            x_train, x_test, y_train, y_test = train_test_split(df.loc[:, ['IW', 'IF', 'VW', 'FP']].values,
                                                                df.loc[:, ['Depth', 'Width']].values)
            pipe = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
            pipe.fit(x_train, y_train)
            s = pickle.dumps(pipe)
            f = open('lr_model', 'wb')
            f.write(s)
            f.close()
            print('Успешно сохранена модель полиноминальной линейной регрессии 2 порядка.')
            print('Метрики точности модели:')
            print(f'\t mean_squared_error: {mean_squared_error(y_true=y_test, y_pred=pipe.predict(x_test))}')
            print(f'\t mean_absolute_error: {mean_absolute_error(y_true=y_test, y_pred=pipe.predict(x_test))}')
            print(f'\t mean_absolute_percentage_error: '
                  f'{mean_absolute_percentage_error(y_true=y_test, y_pred=pipe.predict(x_test))}')
            print(f'\t r2_score: {r2_score(y_true=y_test, y_pred=pipe.predict(x_test))}')
        except Exception as ex:
            print('Произошла ошибка при обработке:')
            print(ex)

    def predict(self, IW, IF, VW, FP):
        [[depth, width]] = self.pipe.predict([[IW, IF, VW, FP]])
        return depth, width

    @staticmethod
    def val_in_range(val_name, val):
        if val < LRPredict.range_limits[val_name][0] or val > LRPredict.range_limits[val_name][1]:
            return False
        return True

    @staticmethod
    def warn_val_in_range(val_name, val):
        if not LRPredict.val_in_range(val_name, val):
            print(f'Введенное значение {val_name} ({LRPredict.range_limits[val_name][0]}'
                  f'-{LRPredict.range_limits[val_name][1]}) выходит за диапазон значений, '
                  f'для которого строилась модель. Результат может быть некорректен.')
