import joblib as jlb
import numpy as np

def predict(data):
    ridge_model = jlb.load('./models/ridge_model.pkl')
    
    coef = ridge_model.coef_
    coef_normalized = coef + (np.abs(coef.min()) + 1)
    
    coef_percent = (coef_normalized/coef_normalized.sum()) * 100
    print(coef)
    
    return ridge_model.predict(data)