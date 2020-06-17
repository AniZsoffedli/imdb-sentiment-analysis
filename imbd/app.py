from flask import Flask, render_template, request
from fastai2.text.all import *
import os

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    context = dict()

    if request.method == 'POST':
        context['method'] = 'POST'
        model_path = os.getcwd() + '/movie_predictor_model/imdb-sample.pkl'
        learn_inf = load_learner(model_path)
        # context['data'] = movie_sentiment_analysis_predict(request.form['user_input'], learn_inf)
        context['data'] = learn_inf.predict(request.form['user_input'])[0].capitalize()

    return render_template('index.html', context=context)


if __name__ == '__main__':
    app.run()

