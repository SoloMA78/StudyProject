# Итоговый проект. Прогноирование ширины и глубины сварного шва

from argparse import ArgumentParser
import pathlib
from src.modeling.LinearRegression import LRPredict
from src.modeling.MLPRegression import MLPPredict

parser = ArgumentParser()
parser.add_argument('-method', nargs=1, choices=['LinearRegression', 'MLPRegression'], default='LinearRegression',
                    help='Метод расчета. Принимает два значения LinearRegression и MLPRegression, '
                         'по-умолчанию - LinearRegression')
parser.add_argument('-mode', nargs=1, choices=['fit', 'predict'], default='predict',
                    help='Режим работы. Принимает два значения fit и predict, по-умолчанию predict.')
parser.add_argument('-data_file_path', nargs=1, type=pathlib.Path, default='./data/raw/ebw_data.csv',
                    help="Путь к файлу с данными  для переобучения модели, по умолчанию ./data/raw/ebw_data.csv.")

# Интрефейс командной строки
if __name__ == '__main__':
    arg_set = parser.parse_args()
    if arg_set.method == 'LinearRegression':
        print('LinearRegression')
        model = LRPredict()
    else:
        print('MLPRegression')
        model = MLPPredict()
    if arg_set.mode == ['fit']:
        model.fit(arg_set.data_file_path)
    else:
        while 1>0:
            print('Прогноз глубины и ширины сварного шва:')
            IW = float(input('\t Введите величину сварочного тока  (IW): '))
            IF = float(input('\t Введите ток фокусировки электронного пучка (IF): '))
            VW = float(input('\t Введите скорость сварки (VW): '))
            FP = float(input('\t Введите расстояние от поверхности образцов до электронно-оптической системы (FP): '))
            model.warn_val_in_range('IW', IW)
            model.warn_val_in_range('IF', IF)
            model.warn_val_in_range('VW', VW)
            model.warn_val_in_range('FP', FP)
            depth, width = model.predict(IW, IF, VW, FP)
            if depth > 0 and width > 0:
                print(f'Прогнозируемая глубина сварного шва {depth:.2f}')
                print(f'Прогнозируемая ширина сварного шва {width:.2f}')
            else:
                print('Использование модели с заданными параметрами дало некорректные результаты')
            if str(input("Следующий прогноз (Y/n)")).lower() == 'n':
                break
    print(arg_set.data_file_path)
    # parser.print_usage()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
