from fastapi import FastAPI, Request
import joblib
import requests
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class CropPredictor:   
    def __init__(self):
        self.numerical_representations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        self.class_labels = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 
                        'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 
                        'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 
                        'rice', 'watermelon']
        self.numerical_to_class = dict(zip(self.numerical_representations, self.class_labels))
        self.loaded_model = joblib.load('Models\decision_tree_model.joblib')
    
    def fahrenheit_to_celsius(self, fahrenheit):
        celsius = fahrenheit - 273.15
        return celsius
    
    def predict_crop(self, N, P, K, pH, city):
        apikey = "68063b330aca634551e488732351a48b"
        base = "https://api.openweathermap.org/data/2.5/weather?q="
        complete = base + city + "&appid=" + apikey
        response = requests.get(complete)
        info = response.json()
        
        temp = info["main"]["temp"]
        celsius_temperature = self.fahrenheit_to_celsius(temp)
        h = info["main"]["humidity"]
        rain = 100  # Modify this if you have a reliable source for rainfall data
        
        input_data = [N, P, K, celsius_temperature, h, pH, rain]
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        train_predictions = self.loaded_model.predict(input_tensor.numpy())
        a = [int(x) for x in train_predictions]
        numerical_representation = a[0]
        label_1 = self.numerical_to_class[numerical_representation]
        
        input_data_2 = [N + 20, P + 20, K + 20, celsius_temperature + 10, h + 20, pH + 3, rain + 60]
        input_tensor_2 = torch.tensor(input_data_2, dtype=torch.float32).unsqueeze(0)
        train_predictions_2 = self.loaded_model.predict(input_tensor_2.numpy())
        numerical_representation_2 = int(train_predictions_2[0])
        label_2 = self.numerical_to_class[numerical_representation_2]
        
        input_data_3 = [N - 5, P - 5, K - 5, celsius_temperature - 3, h - 5, pH - 1, rain - 10]
        input_tensor_3 = torch.tensor(input_data_3, dtype=torch.float32).unsqueeze(0)
        train_predictions_3 = self.loaded_model.predict(input_tensor_3.numpy())
        numerical_representation_3 = int(train_predictions_3[0])
        label_3 = self.numerical_to_class[numerical_representation_3]
        
        return label_1, label_2, label_3

crop_predictor = CropPredictor()

@app.post("/predict/")
async def predict_crop(request: Request):
    data = await request.json()
    N = float(data["N"])
    P = float(data["P"])
    K = float(data["K"])
    pH = float(data["pH"])
    city = data["city"]
    
    recommendation1, recommendation2, recommendation3 = crop_predictor.predict_crop(N, P, K, pH, city)
    return {
        "Prediction_1": recommendation1,
        "Prediction_2": recommendation2,
        "Prediction_3": recommendation3
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)
    #uvicorn crop_api:app --host 127.0.0.1 --port 8003