# app.py
from flask import Flask, render_template, request
import torch
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Initialize the Flask app
app = Flask(__name__)

# Define your PyTorch model architecture here
class DrugSensitivityModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(DrugSensitivityModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load the saved model
input_dim = 611  # Adjust input_dim based on your actual feature count
model = DrugSensitivityModel(input_dim)
model.load_state_dict(torch.load('drug_sensitivity_model.pth'))
model.eval()  # Set to evaluation mode

# Load model columns from CSV file
model_columns = pd.read_csv('model_columns.csv').columns

# Load drug information
drug_info = pd.read_csv('Compounds-annotation.csv')
drug_id_to_name = dict(zip(drug_info['DRUG_ID'], drug_info['DRUG_NAME']))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input_data')
def input_data():
    return render_template('input.html')

@app.route('/drug_lookup')
def drug_lookup():
    # Render the drug lookup template with the drug information
    return render_template('drug_lookup.html', drugs=drug_id_to_name)

@app.route('/predict', methods=['POST'])
def predict():
    # Capture form data
    form_data = request.form.to_dict()

    # Prepare input data for each drug
    drug_ids = [form_data.get(f'DRUG_ID_{i+1}') for i in range(5) if form_data.get(f'DRUG_ID_{i+1}')]
    results = {}

    for drug_id in drug_ids:
        input_data = pd.DataFrame([{
            'AUC': float(form_data['AUC']),
            'Z_SCORE': float(form_data['Z_SCORE']),
            'Whole Exome Sequencing (WES)': form_data['Whole_Exome_Sequencing_WES'],
            'Copy Number Alterations (CNA)': form_data['Copy_Number_Alterations_CNA'],
            'Gene Expression': form_data['Gene_Expression'],
            'DRUG_ID': drug_id,
            'GDSC\nTissue descriptor 1': form_data['GDSC_Tissue_descriptor_1'],
            'GDSC\nTissue\ndescriptor 2': form_data['GDSC_Tissue_descriptor_2']
        }])

        # Apply one-hot encoding to match the model's training data
        input_data = pd.get_dummies(input_data, columns=[
            'Whole Exome Sequencing (WES)',
            'Copy Number Alterations (CNA)',
            'Gene Expression',
            'DRUG_ID',
            'GDSC\nTissue descriptor 1',
            'GDSC\nTissue\ndescriptor 2'
        ])

        # Ensure the DataFrame has all columns expected by the model
        input_data = input_data.reindex(columns=model_columns, fill_value=0)
        input_data = input_data.astype(float)
        input_tensor = torch.tensor(input_data.values).float()

        # Make prediction
        with torch.no_grad():
            ln_ic50 = model(input_tensor).item()

        # Store result
        results[drug_id] = ln_ic50

    # Determine the best drug (lowest IC50)
    best_drug_id = min(results, key=results.get)
    best_drug_name = drug_id_to_name.get(int(best_drug_id), best_drug_id)
    best_ln_ic50 = results[best_drug_id]

    # Generate a plot
    fig, ax = plt.subplots(figsize=(8, 4))
    drug_names = [drug_id_to_name.get(int(drug_id), drug_id) for drug_id in results.keys()]
    ln_ic50_values = list(results.values())
    bars = ax.bar(drug_names, ln_ic50_values, color='skyblue')
    
    # Annotate bars with IC50 values
    for bar, ln_ic50 in zip(bars, ln_ic50_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{ln_ic50:.2f}',
            ha='center',
            va='bottom'
        )

    ax.set_xlabel('Drug')
    ax.set_ylabel('Predicted LN_IC50')
    ax.set_title('Drug Response Prediction')
    
    # Highlight the best suited drug
    ax.bar(best_drug_name, best_ln_ic50, color='orange')

    # Convert plot to PNG image
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches="tight")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Render result page
    return render_template('result.html', plot_url=plot_url, best_drug_name=best_drug_name)
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)
