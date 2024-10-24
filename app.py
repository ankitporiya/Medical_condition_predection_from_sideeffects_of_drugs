# from flask import Flask, render_template, request
# import pandas as pd
# import joblib

# app = Flask(__name__)
# model = joblib.load('model.pkl')

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     drug_name = request.form['drug_name']
#     side_effects = request.form['side_effects']
#     generic_name = request.form['generic_name']
#     drug_classes = request.form['drug_classes']
#     brand_names = request.form['brand_names']
#     activity = request.form['activity']
#     rx_otc = request.form['rx_otc']
#     pregnancy_category = request.form['pregnancy_category']
#     csa = request.form['csa']
#     alcohol = request.form['alcohol']

#     input_data = pd.DataFrame([[drug_name, side_effects, generic_name, drug_classes, brand_names, activity, rx_otc, pregnancy_category, csa, alcohol]],
#                                columns=['drug_name', 'side_effects', 'generic_name', 'drug_classes', 'brand_names', 'activity', 'rx_otc', 'pregnancy_category', 'csa', 'alcohol'])
    
#     input_data_encoded = pd.get_dummies(input_data)
#     prediction = model.predict(input_data_encoded)

#     return render_template('result.html', prediction=prediction[0])

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, render_template, request
# import pandas as pd
# import joblib

# app = Flask(__name__)
# model = joblib.load('model.pkl')

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the input data from the form
#     drug_name = request.form['drug_name']
#     side_effects = request.form['side_effects']
#     generic_name = request.form['generic_name']
#     drug_classes = request.form['drug_classes']
#     brand_names = request.form['brand_names']
#     activity = request.form['activity']
#     rx_otc = request.form['rx_otc']
#     pregnancy_category = request.form['pregnancy_category']
#     csa = request.form['csa']
#     alcohol = request.form['alcohol']

#     # Create a DataFrame with the input data
#     input_data = pd.DataFrame([[drug_name, side_effects, generic_name, drug_classes, brand_names, activity, rx_otc, pregnancy_category, csa, alcohol]],
#                                columns=['drug_name', 'side_effects', 'generic_name', 'drug_classes', 'brand_names', 'activity', 'rx_otc', 'pregnancy_category', 'csa', 'alcohol'])

#     # Load the training data for consistent one-hot encoding
#     training_data = pd.read_csv(r'C:\Users\BE\Desktop\ADS\flask_predictor\drugs_side_effects_drugs_com.csv')  # Use your actual dataset file
#     training_X = training_data[['drug_name', 'side_effects', 'generic_name', 'drug_classes', 'brand_names', 'activity', 'rx_otc', 'pregnancy_category', 'csa', 'alcohol']]
    
#     # Combine the input data and training data for consistent one-hot encoding
#     combined_data = pd.concat([training_X, input_data], ignore_index=True)

#     # One-hot encode the combined data
#     combined_encoded = pd.get_dummies(combined_data)

#     # Separate the input data again
#     input_data_encoded = combined_encoded.iloc[-1:]  # Get the last row (the input data)

#     # Align the columns with the model's expected input
#     model_columns = combined_encoded.columns  # Get the full column names after encoding
#     input_data_encoded = input_data_encoded.reindex(columns=model_columns, fill_value=0)

#     # Make the prediction
#     prediction = model.predict(input_data_encoded)

#     return render_template('result.html', prediction=prediction[0])

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the form
        drug_name = request.form['drug_name']
        side_effects = request.form['side_effects']
        generic_name = request.form['generic_name']
        drug_classes = request.form['drug_classes']
        brand_names = request.form['brand_names']
        activity = request.form['activity']
        rx_otc = request.form['rx_otc']
        pregnancy_category = request.form['pregnancy_category']
        csa = request.form['csa']
        alcohol = request.form['alcohol']

        # Create a DataFrame with the input data
        input_data = pd.DataFrame([[drug_name, side_effects, generic_name, drug_classes, brand_names, activity, rx_otc, pregnancy_category, csa, alcohol]],
                                   columns=['drug_name', 'side_effects', 'generic_name', 'drug_classes', 'brand_names', 'activity', 'rx_otc', 'pregnancy_category', 'csa', 'alcohol'])

        # Load the training data for consistent one-hot encoding
        training_data = pd.read_csv(r'D:\SEM 7\SEM 7\AI-ML(Honars)\project\drugs_side_effects_drugs_com.csv')  # Update with the correct path
        training_X = training_data[['drug_name', 'side_effects', 'generic_name', 'drug_classes', 'brand_names', 'activity', 'rx_otc', 'pregnancy_category', 'csa', 'alcohol']]

        # One-hot encode the training data
        training_encoded = pd.get_dummies(training_X)

        # Combine the input data and training data for consistent one-hot encoding
        combined_data = pd.concat([training_encoded, input_data], ignore_index=True)

        # One-hot encode the combined data again
        combined_encoded = pd.get_dummies(combined_data)

        # Separate the input data again
        input_data_encoded = combined_encoded.iloc[-1:]  # Get the last row (the input data)

        # Align the columns with the model's expected input
        input_data_encoded = input_data_encoded.reindex(columns=training_encoded.columns, fill_value=0)

        # Make the prediction
        prediction = model.predict(input_data_encoded)

        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)




