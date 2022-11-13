# Web backend for model predictions
from flask import Flask, request, render_template
from src.modeling.LinearRegression import LRPredict
from src.modeling.MLPRegression import MLPPredict
import tensorflow

app = Flask(__name__)

lr = LRPredict()
mlp = MLPPredict(vram_lim=True)
param = ('IW', 'IF', 'VW', 'FP')


@app.route('/', methods=['GET', 'POST'])
def start_page():
    mode = request.args.get('mode', 'predict')
    prediction_result = False
    if request.form.get('model'):
        prediction_result = {}
        for key in param:
            try:
                prediction_result[key] = float(request.form.get(key))
                prediction_result['vir_'+key] = LRPredict.val_in_range(key, float(request.form.get(key)))
            except ValueError:
                prediction_result['warn_value_error'] = key
                return render_template('start.html', mode=mode, range_limits=LRPredict.range_limits,
                                       prediction_result=prediction_result)
        model = None
        prediction_result['model'] = request.form.get('model')
        if request.form.get('model') == 'LinearRegression':
            model = lr
        if request.form.get('model') == 'MLPRegression':
            model = mlp
        depth, width = model.predict(IW=prediction_result['IW'],
                                     IF=prediction_result['IF'],
                                     VW=prediction_result['VW'],
                                     FP=prediction_result['FP'])
        prediction_result['Depth'] = f'{depth:.2f}'
        prediction_result['Width'] = f'{width:.2f}'
    return render_template('start.html', mode=mode, range_limits=LRPredict.range_limits,
                           prediction_result=prediction_result)
