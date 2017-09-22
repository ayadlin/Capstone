from flask import Flask
from flask import render_template
from flask import request
import sparse_matrix_functions_web
import graph_maker_web
import predictive_model_web
import pickle
import data_frame_creator
import pandas as pd
import json
from json2html import json2html
from IPython.display import HTML
import random


data  = data_frame_creator.open_pickle('final_data.pickle')
app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

#@app.route('/choose_action', methods=['POST'])
#def choose_action():
#    return 'the answer is' + request.form['choice']

@app.route('/get_evidence', methods=['POST'])
def get_evidence():
    evidence = sparse_matrix_functions_web.provide_evidence(
        gene=request.form['gene'],
        drug=request.form['drug'],
        max_number=int(request.form['max_number']),
        r_s=request.form['r_s'],
        data=data)
    if type(evidence) is not list:
        return "<h3>Either the gene or drug names entered are invalid - download the dictionaries - or there is no evidence of interaction for the requested pair.</h3>"
    result = ''
    df= pd.DataFrame(columns=['paper', 'sentences'])
    evd_list =[]
    for idx, row in enumerate(evidence):
        result += (row[0]+':+\n'
                    +row[1])
        df.loc[idx,'paper'] = row[0]
        df.loc[idx,'sentences'] = row[1]
        evd_dict = {}
        evd_dict["reference"] =row[0]
        evd_dict["sentence"] = row[1]
        evd_list.append(evd_dict)

    inpt = {
        "evidence":evd_list#[{
            #    "rerference":df.loc[:,'paper'].to_json(),
            #    "sentence":df.loc[:,'sentences'].to_json()

        #}]
        }
    #return result


    table = json2html.convert(json=evd_list,escape=False,table_attributes="id=\"info-table\" class=\"table table-bordered table-hover\"")#df.to_json())
    #table = table_raw.replace('"<a', '<a').replace('/a>"', '/a>')

    return render_template('table.html',table=table, title='Evidence')


    #return render_template("templates/evidence.html",table=df.to_html,name ='Evidence')

@app.route('/select_network', methods=['POST'])
def select_network():
    num = random.randint(0,1000000)
    G = graph_maker_web.make_graph_interactive(
        genes=request.form['genes'],
        kind=request.form['kind'],
        path=request.form['path']+str(num),
        lab=request.form['lab'])
    filename = "static/img/"+request.form['path']+str(num)+".png"
    graphname = "static/img/"+request.form['path']+str(num)+".gml"
    ####
    image = "<img src={} height=1000 width=1000>".format(filename)
    graph_link = f'''<a href="{graphname}">Download as .gml</a>'''
    table =  graph_link + '<br/>' + image
    return render_template('table.html',table=table, title='Graph')
    #return "<img src={} height=1000 width=1000>".format(filename)

@app.route('/make_prediction', methods=['POST'])
def make_prediction():
    predicted_drugs = predictive_model_web.provide_drug_predictions(
        genes=request.form['genes'],
        kind=request.form['kind'],
        drug_number=request.form['drug_number'],
        order=request.form['order'])
    if not predicted_drugs:
        return "<h3>Either the gene or drug names entered are invalid - download the dictionaries</h3>"
    if request.form['kind'] in ('r','R'):
        word = 'resistance'
    elif request.form['kind'] in ('s','S'):
        word = 'sensitivity'
    else:
        word = "interactions"
    if request.form['order'] =='a':
        superlative = 'lowest'
    elif request.form['order'] =='d':
        superlative = 'highest'
    caption = "<h3>" + f"Drugs with the {superlative} predicted {word}" + "</h3>"

    table =json2html.convert(json=predicted_drugs,escape=False, table_attributes="id=\"info-table\" class=\"table table-bordered table-hover\"")#['akt1'][0]
    table = caption+table
    return render_template('table.html',table=table, title='Predictions')



if __name__ == "__main__":
    app.run(debug=False)
