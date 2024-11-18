from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.models import vgg19
from torch import nn
from tensorflow.keras.models import load_model
import json
import os
from fastapi import FastAPI, UploadFile, File, Form
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
model = vgg19(pretrained=True)
num_classes = 10
model.classifier[-1] = nn.Linear(4096, num_classes)
model.load_state_dict(torch.load('Models\vgg19_soil_classification.pth', map_location=torch.device('cpu')))
model.eval()

texture_model = load_model('Models\soil.h5')

# Class names and soil information
class_names = ['Alluvial soil', 'Black soil', 'Cinder Soil', 'Clayey soils', 'Laterite soil', 'Loamy soil', 'Peat Soil', 'Sandy loam', 'Sandy soil', 'Yellow Soil']
soil_info = {
    'Alluvial soil': {'sand': 'high', 'silt': 'medium', 'gravel': 'low', 'pH': {'acidic': '7.5', 'neutral': '7', 'alkaline': '6.5'}},
    'Loamy soil': {'sand': 'high', 'silt': 'medium', 'gravel': 'low', 'pH': {'acidic': '6.5', 'neutral': '6.8', 'alkaline': '7.2'}},
    'Black soil': {'sand': 'high', 'silt': 'medium', 'gravel': 'low', 'pH': {'acidic': '8', 'neutral': '7.5', 'alkaline': '7'}},
    'Sandy loam': {'sand': 'high', 'silt': 'medium', 'gravel': 'low', 'pH': {'acidic': '6.8', 'neutral': '7', 'alkaline': '7.5'}},
    'Clayey soils': {'sand': 'high', 'silt': 'medium', 'gravel': 'low', 'pH': {'acidic': '7.2', 'neutral': '6.8', 'alkaline': '6.5'}},
    'Sandy soil': {'sand': 'high', 'silt': 'medium', 'gravel': 'low', 'pH': {'acidic': '6.5', 'neutral': '6.8', 'alkaline': '7.2'}},
    'Laterite soil': {'sand': 'high', 'silt': 'medium', 'gravel': 'low', 'pH': {'acidic': '6.5', 'neutral': '6.8', 'alkaline': '7.2'}},
    'Yellow Soil': {'sand': 'high', 'silt': 'medium', 'gravel': 'low', 'pH': {'acidic': '7', 'neutral': '6.5', 'alkaline': '6'}},
    'Peat Soil': {'sand': 'high', 'silt': 'medium', 'gravel': 'low', 'pH': {'acidic': '5.5', 'neutral': '6', 'alkaline': '6.5'}},
    'Cinder Soil': {'sand': 'high', 'silt': 'medium', 'gravel': 'low', 'pH': {'acidic': '6.5', 'neutral': '6.8', 'alkaline': '7.2'}}
}
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_and_visualize(image):
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = output.max(1)
        predicted_class = class_names[predicted.item()]
    return predicted_class

def make_prediction(image, model):
    img = image.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    classes = ["Gravel", "Sand", "Silt"]
    predicted_value = classes[model.predict(img_batch).argmax()]
    return predicted_value

def classify_images(image, model):
    img = np.array(image)
    img = cv2.resize(img, (1024, 1024))
    im_dim = 256
    gravel_count = sand_count = silt_count = 0
    classes = ['Gravel', 'Sand', 'Silt']

    for r in range(0, img.shape[0], im_dim):
        for c in range(0, img.shape[1], im_dim):
            cropped_img = img[r:r + im_dim, c:c + im_dim]
            if cropped_img.shape[0] == im_dim and cropped_img.shape[1] == im_dim:
                classification = model_classify(cropped_img, model)
                if classification == classes[0]:
                    gravel_count += 1
                elif classification == classes[1]:
                    sand_count += 1
                elif classification == classes[2]:
                    silt_count += 1

    total_count = gravel_count + sand_count + silt_count
    if total_count == 0:
        return [0, 0, 0]
    return [gravel_count / total_count, sand_count / total_count, silt_count / total_count]

def model_classify(cropped_img, model):
    img_array = cropped_img / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    classes = ['Gravel', 'Sand', 'Silt']
    prediction_array = model.predict(img_batch)[0]
    first_idx = np.argmax(prediction_array)
    return classes[first_idx]

def classify_percentage(image):
    proportions = classify_images(image, texture_model)
    return {
        "Gravel": round(proportions[0] * 100, 2),
        "Sand": round(proportions[1] * 100, 2),
        "Silt": round(proportions[2] * 100, 2)
    }

def get_nutrient_and_ph_level(soil_type, texture):
    if (soil_type in soil_info) and (texture in soil_info[soil_type]):
        nutrient_level = soil_info[soil_type][texture]
        ph_level = soil_info[soil_type]['pH']
        return nutrient_level, ph_level
    else:
        return None, None

with open(r'soil_pred_crop\data.json', 'r') as file:
    data = json.load(file)

    
N = 0
P = 0
K = 0

def get_npk_value(location="Coimbatore", level="medium", soil_type='Black soil', texture="sand"):
    global N, P, K
    if level == "Invalid texture":
        return "Invalid texture"
    nutrient_level, ph_level = get_nutrient_and_ph_level(soil_type, texture)
    level_capitalized = nutrient_level.capitalize()

    for entry in data:
        if entry['District'].lower() == location.lower():  # Convert both to lowercase before comparison
            npk_key = f'Nitrogen - {level_capitalized}'
            print(npk_key)
            phosphorous_key = f'Phosphorous - {level_capitalized}'
            potassium_key = f'Potassium - {level_capitalized}'
            N = entry.get(npk_key, "Not available")
            P = entry.get(phosphorous_key, "Not available")
            K = entry.get(potassium_key, "Not available")
            return {
                'Nitrogen': N,
                'Phosphorous': P,
                'Potassium': K
            }
    return "Location not found or level not specified."


def predict_crop(N, P, K, ph_level, location):
    if N == "Not available" or P == "Not available" or K == "Not available":
        return "NPK values are not available for this location and texture."
    else:
        from crop import crop_pred
        crop_instance = crop_pred()
        return crop_instance.crop_pr(N, P, K, ph_level, location)

@app.post("/predict_soil/")
async def predict(image: UploadFile = File(...), location: str = Form(...)):
    try:
        contents = await image.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        soil_type = predict_and_visualize(image)
        
        texture = make_prediction(image, texture_model).lower()
        percent = classify_percentage(image)
        print(percent)
        print(f"Predicted Soil Type: {soil_type}")
        print(f"Predicted Soil Texture: {texture}")
        
        nutrient_level, ph_level = get_nutrient_and_ph_level(soil_type, texture)
        
        if nutrient_level and ph_level:
            print(f"Retrieving NPK values for location: {location} and texture: {texture}")
            npk_values = get_npk_value(location, texture,soil_type, texture)
            
            if npk_values == "Location not found or level not specified.":
                raise HTTPException(status_code=404, detail=f"Location '{location}' not found or level '{texture}' not specified")
            
            print(f"NPK Values: {npk_values}")
            
            N = npk_values['Nitrogen']
            P = npk_values['Phosphorous']
            K = npk_values['Potassium']
            print("pH level:", ph_level) 
            crop = predict_crop(N, P, K, float(ph_level['acidic']), location)
            
            result = {
                "Soil Type": soil_type,
                "Soil Texture": texture,
                "Percentages": percent,
                "Nutrient Level": nutrient_level,
                "pH Level": ph_level,
                "Crop Recommendation": crop
            }
        else:
            result = {"Error": "Invalid soil type or texture."}
        
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
    #uvicorn api:app --reload