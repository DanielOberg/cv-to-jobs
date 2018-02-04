#!/usr/bin/python3

import requests
import json
import pandas as pd
import zipfile
import io
import urllib
import glob
import pypandoc
import os
import time

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors, KDTree

from json.decoder import JSONDecodeError
from flask import Flask, request, redirect, url_for, jsonify, json, send_from_directory, render_template
from werkzeug.utils import secure_filename


def fetch_nr_of_pages():
    HEADERS = {'Accept': 'application/json', 'Accept-Language': 'sv'}
    REQ = requests.get(
        url='http://api.arbetsformedlingen.se/af/v0/platsannonser/matchning?nyckelord=stockholm&antalrader=250',
        headers=HEADERS)
    JSON = REQ.json()
    return JSON['matchningslista']['antal_sidor']


NR_OF_PAGES = fetch_nr_of_pages()
print("Number of pages: " + str(NR_OF_PAGES))


def save_all_jobs(nr_of_pages):
    for i in range(0, nr_of_pages):
        print("page: " + str(i))
        matchingdata = []
        try:
            HEADERS = {'Accept': 'application/json', 'Accept-Language': 'sv'}
            r2 = requests.get(
                url='http://api.arbetsformedlingen.se/af/v0/platsannonser/matchning?nyckelord=stockholm&antalrader=250&sida=' +
                str(i),
                headers=HEADERS)
            list_of_work = r2.json()
            matchingdata = list_of_work['matchningslista']['matchningdata']
        except JSONDecodeError as e:
            continue

        for data in matchingdata:
            if not os.path.isfile('data/' + str(data['annonsid']) + '.json'):
                time.sleep(0.05)
                try:
                    ad = requests.get(
                        url='http://api.arbetsformedlingen.se/af/v0/platsannonser/' +
                        data['annonsid'],
                        headers=HEADERS)
                    ad_json = ad.json()
                    print(ad_json['platsannons']['annons']['yrkesbenamning'])
                    print(ad_json['platsannons']['annons']['annonsid'])
                    print(ad_json['platsannons']['annons']['annonstext'])
                    with open('data/' + str(ad_json['platsannons']['annons']['annonsid']) + '.json', 'w') as file_object:
                        json.dump(
                            ad_json['platsannons']['annons'],
                            file_object,
                            ensure_ascii=False)
                except JSONDecodeError as e:
                    continue
                finally:
                    pass

save_all_jobs(NR_OF_PAGES)


def load_all_jobs_from_disk():
    all_content = {}
    for json_file in glob.glob('data/*.json'):
        print(json_file)
        data = json.load(open(json_file, encoding='utf-8'))
        all_content[data['annonsid']] = data

    df = pd.DataFrame.from_dict(all_content, orient='index')
    return df


dataframe = load_all_jobs_from_disk()


def train_model(df):
    vecto = TfidfVectorizer(
        max_features=None,
        max_df=0.15,
        min_df=2,
        sublinear_tf=True)
    text_clf = Pipeline([('tfidf', vecto)])
    text_clf.fit(df['annonstext'], df['yrkesbenamning'])
    neighbors = NearestNeighbors(
        n_neighbors=10).fit(
        vecto.transform(
            df['annonstext']))
    return (vecto, neighbors)


tfidf, nbrs = train_model(dataframe)

UPLOAD_FOLDER = '/tmp/cvexchange'
ALLOWED_EXTENSIONS = set(['docx', 'odt', 'md', 'txt'])

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/job-from-cv', methods=['POST'])
def job_from_cv():
    text = request.get_data()
    print(text)
    distances, indices = nbrs.kneighbors(tfidf.transform([text]))
    result = []
    for i in indices[0]:
        print(dataframe.iloc[i]['annonsrubrik'] +
              ' - ' + dataframe.iloc[i]['platsannonsUrl'])
        result.append(
            {
                'annonsrubrik': dataframe.iloc[i]['annonsrubrik'],
                'platsannonsUrl': dataframe.iloc[i]['platsannonsUrl']})
    return jsonify(result)


@app.route('/md_from_file', methods=['GET', 'POST'])
def md_from_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            pypandoc.convert_file(filepath, 'md', outputfile=filepath + '.md')
            return send_from_directory(
                app.config['UPLOAD_FOLDER'],
                filename + '.md',
                as_attachment=True)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/', methods=['GET', 'POST'])
def jobs_from_cv():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            pypandoc.convert_file(filepath, 'md', outputfile=filepath + '.md')
            text = open(filepath + '.md', encoding='utf-8').read()
            distances, indices = nbrs.kneighbors(tfidf.transform([text]))
            result = []
            for i in indices[0]:
                result.append(
                    {
                        'annonsrubrik': dataframe.iloc[i]['annonsrubrik'],
                        'platsannonsUrl': dataframe.iloc[i]['platsannonsUrl']})
            return render_template('index.html', entries=result)
    return render_template('index.html', entries=[])



if __name__ == '__main__':
    app.run('0.0.0.0', port=80)
