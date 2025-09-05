# app.py
from flask import Flask, render_template, request
import pickle
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# load model + feature columns
model = pickle.load(open('car_price_model.pkl', 'rb'))
feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))

# load csv for dropdown values (if present)
df = pd.read_csv('car data.csv')

def get_unique(col):
    return sorted(df[col].dropna().unique()) if col in df.columns else []

companies = get_unique('Car_Name')
fuel_types = get_unique('Fuel_Type')
sellers = get_unique('Seller_Type')
transmissions = get_unique('Transmission')
owners = sorted(df['Owner'].dropna().unique()) if 'Owner' in df.columns else [0,1,2]

@app.route('/')
def index():
    current_year = datetime.now().year
    years = list(range(current_year, current_year-30, -1))
    return render_template('index.html',
                           companies=companies,
                           fuel_types=fuel_types,
                           sellers=sellers,
                           transmissions=transmissions,
                           owners=owners,
                           years=years)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # read form
        company = request.form.get('company')
        year = int(request.form.get('year'))
        kms_driven = float(request.form.get('kms_driven'))
        fuel_type = request.form.get('fuel_type')
        seller_type = request.form.get('seller_type')
        transmission = request.form.get('transmission')
        owner = int(request.form.get('owner'))

        # compute Age same way as train script
        current_year = datetime.now().year
        age = current_year - year

        # create input dict matching column names used during training
        input_dict = {}
        for col in feature_columns:
            if col == 'Kms_Driven':
                input_dict[col] = kms_driven
            elif col == 'Owner':
                input_dict[col] = owner
            elif col == 'Age':
                input_dict[col] = age
            elif col == 'Car_Name':
                input_dict[col] = company
            elif col == 'Fuel_Type':
                input_dict[col] = fuel_type
            elif col == 'Seller_Type':
                input_dict[col] = seller_type
            elif col == 'Transmission':
                input_dict[col] = transmission
            else:
                # default safe fallback
                input_dict[col] = 0

        input_df = pd.DataFrame([input_dict], columns=feature_columns)

        pred = model.predict(input_df)[0]
        # NOTE: prediction is in same units as your dataset target (e.g., lakhs)
        return render_template('index.html', prediction_text=f'Predicted price: {round(pred, 2)}',
                               companies=companies, fuel_types=fuel_types, sellers=sellers,
                               transmissions=transmissions, owners=owners)
    except Exception as e:
        return render_template('index.html', error=str(e),
                               companies=companies, fuel_types=fuel_types, sellers=sellers,
                               transmissions=transmissions, owners=owners)

if __name__ == '__main__':
    app.run(debug=True)
