import io
import json
import requests
import textwrap
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, validator
from typing import List
from googletrans import Translator
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pest_names = [
    "rice leaf roller", "rice leaf caterpillar", "paddy stem maggot", "asiatic rice borer",
    "yellow rice borer", "rice gall midge", "Rice Stemfly", "brown plant hopper",
    "white backed plant hopper", "small brown plant hopper", "rice water weevil", "rice leafhopper",
    "grain spreader thrips", "rice shell pest", "grub", "mole cricket",
    "wireworm", "white margined moth", "black cutworm", "large cutworm",
    "yellow cutworm", "red spider", "corn borer", "army worm",
    "aphids", "Potosiabre vitarsis", "peach borer", "english grain aphid",
    "green bug", "bird cherry-oataphid", "wheat blossom midge", "penthaleus major",
    "longlegged spider mite", "wheat phloeothrips", "wheat sawfly", "cerodonta denticornis",
    "beet fly", "flea beetle", "cabbage army worm", "beet army worm",
    "Beet spot flies", "meadow moth", "beet weevil", "sericaorient alismots chulsky",
    "alfalfa weevil", "flax budworm", "alfalfa plant bug", "tarnished plant bug",
    "Locustoidea", "lytta polita", "legume blister beetle", "blister beetle",
    "therioaphis maculata Buckton", "odontothrips loti", "Thrips", "alfalfa seed chalcid",
    "Pieris canidia", "Apolygus lucorum", "Limacodidae", "Viteus vitifoliae",
    "Colomerus vitis", "Brevipoalpus lewisi McGregor", "oides decempunctata", "Polyphagotars onemus latus",
    "Pseudococcus comstocki Kuwana", "parathrene regalis", "Ampelophaga", "Lycorma delicatula",
    "Xylotrechus", "Cicadella viridis", "Miridae", "Trialeurodes vaporariorum",
    "Erythroneura apicalis", "Papilio xuthus", "Panonchus citri McGregor", "Phyllocoptes oleiverus ashmead",
    "Icerya purchasi Maskell", "Unaspis yanonensis", "Ceroplastes rubens", "Chrysomphalus aonidum",
    "Parlatoria zizyphus Lucus", "Nipaecoccus vastalor", "Aleurocanthus spiniferus", "Tetradacus c Bactrocera minax",
    "Dacus dorsalis(Hendel)", "Bactrocera tsuneonis", "Prodenia litura", "Adristyrannus",
    "Phyllocnistis citrella Stainton", "Toxoptera citricidus", "Toxoptera aurantii", "Aphis citricola Vander Goot",
    "Scirtothrips dorsalis Hood", "Dasineura sp", "Lawana imitata Melichar", "Salurnis marginella Guerr",
    "Deporaus marginatus Pascoe", "Chlumetia transversa", "Mango flat beak leafhopper", "Rhytidodera bowrinii white",
    "Sternochetus frigidus", "Cicadellidae"
]
num_classes = len(pest_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loaded_model = models.resnet50(pretrained=False)
loaded_model.fc = nn.Linear(loaded_model.fc.in_features, num_classes)
loaded_model.load_state_dict(torch.load('Models\resnet50_0.497.pkl', map_location=device))
loaded_model.to(device)
loaded_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def search_pesticides(pest_name, number):
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "redirects": number,
            "titles": pest_name
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        extract = f"Pest Name:{pest_name} "
        page_id = next(iter(data['query']['pages'].keys()))
        extract += data['query']['pages'][page_id].get('extract', 'No extract available.')
        return extract
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving data for {pest_name}: {e}")
        return None


def display_search_results(pest_info):
    if pest_info:
        return f"<p>{pest_info}</p>"
    else:
        return "<p>No information found.</p>"


@app.post("/pest_type/")
async def pest_type(file: UploadFile = File(...)):
    global pest
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB") 
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = loaded_model(image)
            _, predicted_class = torch.max(output, 1)
        predicted_label = pest_names[predicted_class.item()]
        pest = predicted_label

        search_results = search_pesticides(predicted_label, 5)

        html_response = HTMLResponse(content=display_search_results(search_results))
        return html_response

    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Error processing image")

class QueryRequest(BaseModel):
    queries: List[str]
    language: str

    @validator('language')
    def validate_language(cls, value):
        if len(value) != 2:
            raise ValueError("Language code must be exactly two characters long")
        return value.lower()

@app.post("/query/")
async def query(request: QueryRequest):
    try:
        translator = Translator()
        answers = []
        target_language = request.language
        
        paragraphs = search_pesticides(pest, 10) if pest else "No pest information available."

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_text(paragraphs)

        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        
        for query in request.queries:
            translated_query = translator.translate(query, src=target_language, dest='en').text
            inputs = tokenizer.encode(f"answer the question: {translated_query} context: {str(docs[0])}", return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(inputs, max_length=150, num_return_sequences=1, early_stopping=True)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated_answer = translator.translate(answer, src='en', dest=target_language).text
            answers.append(translated_answer)

        return {"answers": answers}
    except Exception as e:
        print("Error processing query:", e)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
#uvicorn pest:app --host 127.0.0.1 --port 8002