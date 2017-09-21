from flask import Flask
from flask import render_template
from flask import request
import sparse_matrix_functions_web
import graph_maker
import predictive_model
import pickle
import data_frame_creator


data  = data_frame_creator.open_pickle('final_data.pickle')
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('inputs.html')

@app.route('/choose_action', methods=['POST'])
def choose_action():
    return 'the answer is' + request.form['choice']


@app.route('/get_evidence', methods=['POST'])
def get_evidence():
    evidence = sparse_matrix_functions_web.provide_evidence(
        gene=request.form['gene'],
        drug=request.form['drug'],
        max_number=request.form['max_number'],
        r_s=request.form['r_s'],
        data=data)
    return evidence






if __name__ == "__main__":
    app.run(debug=True)
