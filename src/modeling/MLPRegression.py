# MLPRegressor

import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, UnitNormalization, Dropout, Normalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, SGD
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from pathlib import Path
import tensorflow as tf

class MLPPredict:
    range_limits = {
        'IW': [43, 49],
        'IF': [131, 150],
        'VW': [4.5, 12],
        'FP': [50, 125]
    }

    def __init__(self, vram_lim=False):
        if vram_lim:
            # Устанавливаем лимит видеопамяти при работе с Flask
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
                except RuntimeError as e:
                    print(e)
        self.model = MLPPredict.get_model()
        if Path('model/MLP_model.hd5').exists():
            self.model = load_model('model/MLP_model.hd5')
            print('Model loaded')
        else:
            self.model = MLPPredict.get_model()
        self.model.compile()

    @staticmethod
    def get_model():
        model = Sequential()
        model.add(UnitNormalization())
        model.add(Dense(50, activation='relu'))
        #model.add(Normalization())
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
            model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-02))
            checkpoint_filepath = 'model/tmp/'
            callback = [EarlyStopping(monitor='val_loss', patience=50, min_delta=1e-09, verbose=1),
                        ModelCheckpoint(
                            filepath=checkpoint_filepath,
                            save_weights_only=True,
                            monitor='val_loss',
                            mode='min',
                            save_best_only=True)
                        ]
            history = model.fit(train_ds.loc[:, ['IW', 'IF', 'VW', 'FP']].values, train_ds.loc[:, ['Depth', 'Width']].values,
                      validation_split=0.2, epochs=50000,
                      callbacks=[callback], verbose=1, shuffle=True)
            model.load_weights(checkpoint_filepath)
            model.save('model/MLP_model.hd5', save_format='h5')
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
