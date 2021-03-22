from gensim.models.doc2vec import Doc2Vec
from flask import Flask, request
from flask_cors import CORS, cross_origin
import requests
import xml.etree.ElementTree as ET
import json
import settings

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-TYpe'

model = Doc2Vec.load('doc2vec.model')
yahoo_url = 'https://jlp.yahooapis.jp/MAService/V1/parse'
client_id = settings.ID

@app.route("/", methods=['POST'])
@cross_origin()
def post():
    input_data = request.json
    sentence = input_data['text']
    word_list = []

    data = {
        'appid': client_id,
        'results': 'ma',
        'sentence': sentence
    }

    res = requests.post(yahoo_url, data=data)

    root = ET.fromstring(res.text)

    for e in root.getiterator('{urn:yahoo:jp:jlp}surface'):
        word_list.append(e.text)

    topn = input_data['topn']
    result = model.docvecs.most_similar([model.infer_vector(word_list)], topn=topn)
    print(result)

    dict_data = {}
    for i in range(topn):
        # key = str(i+1) + '位'
        company_key = result[i][0]
        dict_data[company_key] = result[i][1]

    json_data = json.dumps(dict_data, ensure_ascii=False, indent=2)
    print(json_data)
    return json_data

if __name__ == '__main__':
    app.run()
