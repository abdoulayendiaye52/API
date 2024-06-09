# Importer les bibliothèques nécessaires
import uvicorn  # For ASGI
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

# Définir le modèle de données pour l'entrée
class BankCustomerData(BaseModel):
    CreditScore: float
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Loyalty: float
    Geography_Germany: int
    Geography_Spain: int
    Geography_France: int

# Créer l'objet de l'application et charger le modèle de classification
app = FastAPI()
with open('classifier.pkl', 'rb') as pickle_in:
    classifier = pickle.load(pickle_in)

# Route d'index
@app.get('/')
def index():
    return {'message': 'Hello, stranger'}

# Route avec un paramètre
@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Welcome {name}'}

# Fonctionnalité de prédiction
@app.post('/predict')
def predict_churn(data: BankCustomerData):
    data_dict = data.dict()

    CreditScore = data_dict['CreditScore']
    Balance = data_dict['Balance']
    NumOfProducts = data_dict['NumOfProducts']
    HasCrCard = data_dict['HasCrCard']
    IsActiveMember = data_dict['IsActiveMember']
    EstimatedSalary = data_dict['EstimatedSalary']
    Loyalty = data_dict['Loyalty']
    Geography_Germany = data_dict['Geography_Germany']
    Geography_Spain = data_dict['Geography_Spain']
    Geography_France = data_dict['Geography_France']
    
    prediction = classifier.predict([[CreditScore, Balance, NumOfProducts, HasCrCard, IsActiveMember,
                                      EstimatedSalary, Loyalty, Geography_Germany, Geography_Spain, Geography_France]])

    if prediction[0] == 1:
        prediction_text = 'Customer will leave the Bank'
    else:
        prediction_text = 'Customer will continue with the Bank'

    return {'Prediction Text': prediction_text}

# Exécuter l'API avec uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8008)
