# Sample participant submission for testing
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import random
import pickle
app = Flask(__name__)



class Solution:
    def __init__(self):
        #Initialize any global variables here
        with open('cbclass_best.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        self.model = loaded_model
#         with open('pca.pkl', 'rb') as file:
#             pca = pickle.load(file)

#         self.pca = pca

    def calculate_death_prob(self, timeknown, cost, reflex, sex, blood, bloodchem1, bloodchem2, temperature, race,
                             heart, psych1, glucose, psych2, dose, psych3, bp, bloodchem3, confidence, bloodchem4,
                             comorbidity, totalcost, breathing, age, sleep, dnr, bloodchem5, pdeath, meals, pain,
                             primary, psych4, disability, administratorcost, urine, diabetes, income, extraprimary,
                             bloodchem6, education, psych5, psych6, information, cancer):
        fill_dict = {'timeknown': 485.55993199206574,
                     'reflex': 9.995841085100738,
                     'blood': 12.428563615313905,
                     'bloodchem2': 1.7522679232706015,
                     'temperature': 37.3315513589643,
                     'heart': 97.23968521436669,
                     'psych1': 1.8709266081042788,
                     'bp': 84.57523377727402,
                     'confidence': 12.35524644813197,
                     'comorbidity': 1.8641258146783792,
                     'totalcost': 30804.339769989012,
                     'breathing': 23.264522527628223,
                     'age': 66.54257906309152,
                     'meals': 137.53060357041656,
                     'pain': 2.999574950410881,
                     'administratorcost': 59757.22219786097,
                     'psych5': 22.542951563589593,
                     'psych6': 1.88585429325588,
                     'information': 14.720073926642025,
                     'education': 13.085434938782324,
                     'sex': 'female',
                     'race': 'white',
                     'dnr': 'no dnr',
                     'cancer': 'no',
                     'extraprimary': 'ARF/MOSF',
                     'diabetes': 0.0,
                     'primary': 'ARF/MOSF w/Sepsis',
                     'income': 'under $11k'}
        
        numeric_columns = ['timeknown','reflex','blood','bloodchem2','temperature','heart','psych1','bp','confidence','comorbidity','totalcost','breathing','age','meals','pain','administratorcost','psych5','psych6','information','education']
        categorical_columns = ['sex','race','dnr','cancer','extraprimary','diabetes','primary','income']

        labels = numeric_columns+categorical_columns
        
        values = [x for x in [timeknown, reflex, blood, bloodchem2, temperature, heart, psych1, bp, confidence, comorbidity, totalcost, breathing, age, meals, pain, administratorcost, psych5, psych6, information,education, sex, race, dnr, cancer, extraprimary, diabetes, primary,income]]
        new_val = []
        for val in values:
            if val=='nan':
                new_val.append(None)
            else:
                new_val.append(val)
        values = new_val
        df = dict()
        for label, value in zip(labels, values):
            df[label] = [value]
        df = pd.DataFrame(df)
        for col in numeric_columns+categorical_columns:
            df[col].fillna(value=fill_dict[col],inplace=True)
            
        df['sex'] = df['sex'].replace({'male': 1,'M': 1, '1': 1,'Male':1, 'female': 0})
        df['race'] = df['race'].replace({'white': '1','black': '0','hispanic':'2','other':'4','asian':'3'})
        df['cancer'] = df['cancer'].replace({'yes': '1','no': '0','metastatic':'2'})
        df['dnr'] = df['dnr'].replace({'dnr before sadm': '0','no dnr': '1','dnr after sadm':'2'})
        df['primary'] = df['primary'].replace({'Cirrhosis': '0','Colon Cancer': '1','ARF/MOSF w/Sepsis':'2', 'COPD':'3','Lung Cancer':'4','MOSF w/Malig':'5', 'CHF':'6','Coma':'7'})
        df['extraprimary'] = df['extraprimary'].replace({'COPD/CHF/Cirrhosis': '0','Cancer': '1','ARF/MOSF':'2','Coma':'3'})
        df['income'] = df['income'].replace({'under $11k':0,'$11-$25k':1,'$25-$50k':2,'>$50k':3})
        df = df[numeric_columns+categorical_columns]
        df.replace('', 0, inplace=True)
        df.fillna(0, inplace=True)
#         df = df.drop(numeric_columns, axis=1)
        inp = df.values
        n = inp.shape[1]
        return self.model.predict_proba(inp.reshape(1,n))[0][0]

    
    
# BOILERPLATE
@app.route("/death_probability", methods=["POST"])
def q1():
    solution = Solution()
    data = request.get_json()
    return {
        "probability": solution.calculate_death_prob(data['timeknown'], data['cost'], data['reflex'], data['sex'], data['blood'],
                                            data['bloodchem1'], data['bloodchem2'], data['temperature'], data['race'],
                                            data['heart'], data['psych1'], data['glucose'], data['psych2'],
                                            data['dose'], data['psych3'], data['bp'], data['bloodchem3'],
                                            data['confidence'], data['bloodchem4'], data['comorbidity'],
                                            data['totalcost'], data['breathing'], data['age'], data['sleep'],
                                            data['dnr'], data['bloodchem5'], data['pdeath'], data['meals'],
                                            data['pain'], data['primary'], data['psych4'], data['disability'],
                                            data['administratorcost'], data['urine'], data['diabetes'], data['income'],
                                            data['extraprimary'], data['bloodchem6'], data['education'], data['psych5'],
                                            data['psych6'], data['information'], data['cancer'])}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)
