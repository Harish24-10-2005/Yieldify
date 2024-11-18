from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import requests
from transformers import T5ForConditionalGeneration, T5Tokenizer
from googletrans import Translator
from pydantic import BaseModel, validator
from typing import List
import torch.nn as nn
import torchvision.models as models
import io
import os
from langchain.text_splitter import CharacterTextSplitter
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = 'Models\cnn_model.pth'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
cnn_model = models.densenet121(pretrained=True)
cnn_model.classifier = nn.Linear(cnn_model.classifier.in_features, len(checkpoint['class_to_index']))
cnn_model.load_state_dict(checkpoint['model_state_dict'])
cnn_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

labels = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

plant = None

def search_pesticides(predicted_class):
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": f"{predicted_class} disease treatment",
            "srprop": "snippet",
            "srlimit": "5"
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        search_results = data['query']['search']
        return search_results
    except requests.exceptions.RequestException as e:
        print("Error retrieving data:", e)
        return None

def display_search_results(search_results):
    if not search_results:
        return "<p>No search results found.</p>"
    result_text = "<h3>About and Cure steps:</h3>"
    for result in search_results:
        title = result['title']
        snippet = result['snippet'].replace('<span class="searchmatch">', '').replace('</span>', '')
        result_text += f"<p><strong>{title}:</strong><br>{snippet}</p>"
    return result_text

class QueryRequest(BaseModel):
    queries: List[str]
    language: str

    @validator('language')
    def validate_language(cls, value):
        if len(value) != 2:
            raise ValueError("Language code must be exactly two characters long")
        return value.lower()

@app.post("/predict/")
async def predict_plant_disease(file: UploadFile = File(...)):
    global plant
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = cnn_model(image)
            _, predicted = torch.max(output, 1)
        predicted_label = labels[predicted.item()]
        plant = predicted_label
        search_results = search_pesticides(predicted_label)
        display_results = display_search_results(search_results)
        return {
            "predicted_label": predicted_label,
            "search_results": display_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting plant disease: {str(e)}")

@app.post("/answer/")
async def query(request: QueryRequest):
    try:
        if not plant:
            raise HTTPException(status_code=400, detail="Plant disease not predicted yet.")
        translator = Translator()
        answers = []
        target_language = request.language
        file_path = f"Plant disease data/{plant}.txt"

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Data file for {plant} not found.")

        with open(file_path, 'r') as file:
            document = file.read()

        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.split_text(document)

        for query in request.queries:
            translated_query = translator.translate(query, src=target_language, dest='en').text
            answer_sentences = []
            for doc in docs:
                inputs = tokenizer.encode(f"answer the question: {translated_query} context: {doc}", return_tensors="pt", max_length=512, truncation=True)
                outputs = model.generate(inputs, max_length=512, num_return_sequences=1, early_stopping=True)
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if answer.strip():
                    translated_answer = translator.translate(answer, src='en', dest=target_language).text
                    answer_sentences.append(translated_answer)
            if answer_sentences:
                answers.append(" ".join(answer_sentences))

        if not answers:
            answers.append("No suitable answer found.")

        return {
            "answers": answers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
 #   uvicorn plant:app --host 127.0.0.1 --port 8001
