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
        with open('RF.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        self.model = loaded_model
        with open('pca.pkl', 'rb') as file:
            pca = pickle.load(file)

        self.pca = pca

    def calculate_death_prob(self, timeknown, cost, reflex, sex, blood, bloodchem1, bloodchem2, temperature, race,
                             heart, psych1, glucose, psych2, dose, psych3, bp, bloodchem3, confidence, bloodchem4,
                             comorbidity, totalcost, breathing, age, sleep, dnr, bloodchem5, pdeath, meals, pain,
                             primary, psych4, disability, administratorcost, urine, diabetes, income, extraprimary,
                             bloodchem6, education, psych5, psych6, information, cancer):
        


        labels = ['timeknown', 'reflex', 'sex', 'blood', 'bloodchem2', 'temperature',
       'race', 'heart', 'psych1', 'dose', 'bp', 'confidence', 'comorbidity',
       'totalcost', 'breathing', 'age','dnr', 'meals', 'pain','primary',
       'administratorcost', 'diabetes','extraprimary', 'psych5', 'psych6', 'information',
       'cancer']
        if sex in ['male','Male','M']:
            sex = 1
        elif sex == 'female':
            sex = 0
        

      
    #Replacing null with mean
        if blood is None:
            blood= 12.428563615313905
        if information is None:
            information= 14.746332255310811
        if bloodchem2 is None:
            bloodchem2= 1.7536666942161871
        if confidence is None:
            confidence= 12.374289533017738   
        if totalcost is None:
            totalcost= 30830.356076021468   
        if administratorcost is None:
            administratorcost= 59874.110327596536   
        if psych5 is None:
            psych5= 22.543661179523014
        
      #Replacing null with mode
        if race is None:
            race= 'white'
        if dnr is None:
            dnr= 'no dnr'
      
        
#replace 
        if race == 'white':
            race=1
        elif race == 'black':
            race=0
        elif race == 'hispanic':
            race=2
        elif race == 'other':
            race=4
        elif race == 'asian':
            race=3
         
        cancer_dict = {
            'yes':1,
            'no':0,
            'metastatic':2
        }
        cancer = cancer_dict[cancer]
        
        
      #  ['dnr before sadm', 'no dnr', 'dnr after sadm', nan],
        if dnr=='dnr before sadm':
            dnr=0
        elif dnr=='no dnr':
            dnr=1
        elif dnr=='dnr after sadm':
            dnr=2
            
#             f['primary'] = df['primary'].replace({'Cirrhosis': '0','Colon Cancer': '1','ARF/MOSF w/Sepsis':'2', 'COPD':'3','Lung Cancer':'4','MOSF w/Malig':'5', 'CHF':'6','Coma':'7'})

        if primary=='Cirrhosis':
            primary=0
        elif primary=='Colon Cancer':
            primary=1
        elif primary=='ARF/MOSF w/Sepsis':
            primary=2
        elif primary=='COPD':
            primary=3
        elif primary=='Lung Cancer':
            primary=4
        elif primary=='MOSF w/Malig':
            primary=5
        elif primary=='CHF':
            primary=6
        elif primary=='Coma':
            primary=7          
#   df['extraprimary'] = df['extraprimary'].replace({'COPD/CHF/Cirrhosis': '0','Cancer': '1','ARF/MOSF':'2','Coma':'3'})
        
        if extraprimary=='COPD/CHF/Cirrhosis':
            extraprimary=0
        elif extraprimary=='Cancer':
            extraprimary=1
        elif extraprimary=='ARF/MOSF':
            extraprimary=2
        elif extraprimary=='Coma':
            extraprimary=3
            
        
        inp = [timeknown ,  reflex ,  sex ,  blood ,  bloodchem2 ,  temperature , race ,  heart ,  psych1 ,  dose ,  bp ,  confidence ,  comorbidity , totalcost ,  breathing ,  age , dnr, meals ,  pain , primary, administratorcost ,  diabetes ,extraprimary,psych5 ,  psych6 ,  information , cancer]
        values = [float(x) for x in inp]
        df = dict()
        for label, value in zip(labels, values):
            df[label] = [value]
        df = pd.DataFrame(df)
        df.replace('', 0, inplace=True)
        df.fillna(0, inplace=True)
        
        
        numeric_cols=['timeknown','reflex','blood','bloodchem2','temperature','heart','psych1','bp','confidence','comorbidity','totalcost','breathing','age','meals','administratorcost','psych5','psych6','information']
            
        X_train=df[numeric_cols].values
        X_train=self.pca.transform(X_train)
        for i in range(X_train.shape[1]):
            df[str(i)+'_pca']=X_train[:,i]
        df.drop(numeric_cols,axis=1,inplace=True)
        
        inp = df.values
        return self.model.predict_proba(inp.reshape(1,15))[0][0]

    
    
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
    app.run(host="0.0.0.0", port=5566)
