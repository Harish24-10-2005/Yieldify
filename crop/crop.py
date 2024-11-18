class crop_pred:   
    def crop_pr(self,N,P,K,pH,city):
        import torch
        import joblib
        import requests,json
        numerical_representations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        class_labels = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 
                        'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 
                        'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 
                        'rice', 'watermelon']
        numerical_to_class = dict(zip(numerical_representations, class_labels))
        loaded_model = joblib.load('decision_tree_model.joblib')
        
        apikey = "68063b330aca634551e488732351a48b"
        base = "https://api.openweathermap.org/data/2.5/weather?q="
        complete = base + city + "&appid="+apikey
        response = requests.get(complete)
        info= response.json()
        def fahrenheit_to_celsius(fahrenheit):
            celsius = fahrenheit - 273.15
            return celsius
            
        temp=info["main"]["temp"]
        celsius_temperature = fahrenheit_to_celsius(temp)
        h = info["main"]["humidity"]
        import random
        rain = random.randint(50, 200)
        
        input_data = [N,P,K, celsius_temperature, h, pH,	rain]
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        train_predictions = loaded_model.predict(input_tensor.numpy())
        a = [int(x) for x in train_predictions]
        numerical_representation = a[0]
        label_1 = numerical_to_class[numerical_representation]
        print("Predicted Crop Recommendation->1:",label_1)
        
        input_data_2 = [N + 20, P + 20, K + 20, celsius_temperature + 10, h + 20, pH + 3, rain + 60]
        input_tensor_2 = torch.tensor(input_data_2, dtype=torch.float32).unsqueeze(0)
        train_predictions_2 = loaded_model.predict(input_tensor_2.numpy())
        numerical_representation_2 = int(train_predictions_2[0])
        label_2 = numerical_to_class[numerical_representation_2]
        
        print("Predicted Crop Recommendation->2:", label_2)
        
        
        input_data_3 = [N - 5, P - 5, K - 5, celsius_temperature - 3, h - 5, pH - 1, rain - 10]
        input_tensor_3 = torch.tensor(input_data_3, dtype=torch.float32).unsqueeze(0)
        train_predictions_3 = loaded_model.predict(input_tensor_3.numpy())
        numerical_representation_3 = int(train_predictions_3[0])
        label_3 = numerical_to_class[numerical_representation_3]
        
        print("Predicted Crop Recommendation->3:", label_3)