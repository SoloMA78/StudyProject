# MLPRegressor

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, UnitNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, SGD
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from pathlib import Path

class MLPPredict:
    range_limits = {
        'IW': [43, 49],
        'IF': [131, 150],
        'VW': [4.5, 12],
        'FP': [50, 125]
    }

    def __init__(self):
        self.model = MLPPredict.get_model()
        if Path('MLP_model').exists():
            self.model.load_weights('MLP_model')
        self.model.compile()

    @staticmethod
    def get_model():
        model = Sequential()
        model.add(UnitNormalization())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(2))
        return model

    @staticmethod
    def fit(data_path):
        try:
            df = pd.read_csv(data_path)
            train_ds = df.sample(frac=0.9)
            test_ds = df.drop(train_ds.index)
            model = MLPPredict.get_model()
            model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-03))
            checkpoint_filepath = '/tmp/checkpoint'
            callback = [EarlyStopping(monitor='val_loss', patience=15, min_delta=1e-09),
                        ModelCheckpoint(
                            filepath=checkpoint_filepath,
                            save_weights_only=True,
                            monitor='val_loss',
                            mode='min',
                            save_best_only=True)
                        ]
            history = model.fit(train_ds.loc[:, ['IW', 'IF', 'VW', 'FP']].values, train_ds.loc[:, ['Depth', 'Width']].values,
                      validation_split=0.2, epochs=50000,
                      callbacks=[callback], verbose=0, shuffle=True)
            model.load_weights(checkpoint_filepath)
            model.save('MLP_model')
            print(f'Успешно сохранена модель нейронной сети. Эпох обучения до остановки {history.epoch[-1]}')
            y_pred = model.predict(test_ds.loc[:, ['IW', 'IF', 'VW', 'FP']].values, verbose=0)
            y_true = test_ds.loc[:, ['Depth', 'Width']].values
            print('Метрики точности модели на тестовой выборке:')
            mse_test = mean_squared_error(y_true=y_true, y_pred=y_pred)
            mae_test = mean_absolute_error(y_true=y_true, y_pred=y_pred)
            mape_test = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
            r2 = r2_score(y_true=y_true, y_pred=y_pred)
            print(f'\t mean_squared_error: {mse_test:.4f}')
            print(f'\t mean_absolute_error: {mae_test:.4f}')
            print(f'\t mean_absolute_percentage_error: {mape_test:.4f}')
            print(f'\t r2: {r2:.4f}')
        except Exception as ex:
            print('Произошла ошибка при обработке:')
            print(ex)

    def predict(self, IW, IF, VW, FP):
        [[depth, width]] = self.model.predict([[IW, IF, VW, FP]])
        return depth, width

    @staticmethod
    def val_in_range(val_name, val):
        if val < MLPPredict.range_limits[val_name][0] or val > MLPPredict.range_limits[val_name][1]:
            return False
        return True

    @staticmethod
    def warn_val_in_range(val_name, val):
        if not MLPPredict.val_in_range(val_name, val):
            print(f'Введенное значение {val_name} ({MLPPredict.range_limits[val_name][0]}'
                  f'-{MLPPredict.range_limits[val_name][1]}) выходит за диапазон значений, '
                  f'для которого строилась модель. Результат может быть некорректен.')
