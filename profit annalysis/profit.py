from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

days_to_yield = {
    "Apple": 365,
    "Banana": 300,
    "Blackgram": 65,
    "Chickpea": 95,
    "Coconut": 365,
    "Coffee": 365,
    "Cotton": 165,
    "Grapes": 365,
    "Jute": 135,
    "Kidneybeans": 90,
    "Lentil": 100,
    "Maize": 105,
    "Mango": 365,
    "Mothbeans": 60,
    "Mungbean": 65,
    "Muskmelon": 82,
    "Orange": 365,
    "Papaya": 210,
    "Pigeonpeas": 165,
    "Pomegranate": 365,
    "Rice": 105,
    "Watermelon": 82
}

growth_periods = {
    "Apple": "January to December",
    "Banana": "January to December",
    "Blackgram": "June to November",
    "Chickpea": "October to February",
    "Coconut": "January to December",
    "Coffee": "January to December",
    "Cotton": "May to November",
    "Grapes": "January to December",
    "Jute": "March to August",
    "Kidneybeans": "June to November",
    "Lentil": "October to February",
    "Maize": "June to October",
    "Mango": "January to June",
    "Mothbeans": "May to September",
    "Mungbean": "June to October",
    "Muskmelon": "March to July",
    "Orange": "January to December",
    "Papaya": "January to December",
    "Pigeonpeas": "June to December",
    "Pomegranate": "January to December",
    "Rice": "June to November",
    "Watermelon": "March to July"
}

class CropRequest(BaseModel):
    crop: str
    seeding_date: datetime

@app.post("/profitability")
def is_profit(request: CropRequest):
    crop = request.crop.capitalize()
    seeding_date = request.seeding_date

    if crop not in days_to_yield:
        raise HTTPException(status_code=400, detail="Invalid crop name")
    
    days_to_grow = days_to_yield[crop]
    yield_date = seeding_date + timedelta(days=days_to_grow)
    start_month, end_month = growth_periods[crop].split(" to ")
    start_month_index = datetime.strptime(start_month, "%B").month
    end_month_index = datetime.strptime(end_month, "%B").month
    
    diff_start_month = abs(seeding_date.month - start_month_index)
    diff_end_month = abs(seeding_date.month - end_month_index)
    
    if diff_start_month == 0 or diff_end_month == 0:
        return f"Your crop {crop} gets high profit when you seed it on {seeding_date.strftime('%Y-%m-%d')}"
    elif diff_start_month == 1 or diff_end_month == 1:
        return f"Your crop {crop} gets medium profit when you seed it on {seeding_date.strftime('%Y-%m-%d')}"
    elif diff_start_month == 2 or diff_end_month == 2:
        return f"Your crop {crop} gets less profit when you seed it on {seeding_date.strftime('%Y-%m-%d')}"
    else:
        return f"Your crop {crop} doesn't yield profit when you seed it on {seeding_date.strftime('%Y-%m-%d')}"
#uvicorn profit:app --host 127.0.0.1 --port 8007