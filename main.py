from fastapi import FastAPI
import pickle
from pydantic import BaseModel

class Data(BaseModel):
    variance:float
    skewness:float
    curtosis:float
    entropy:float

file=open("classifier.pkl",'rb')
model=pickle.load(file)

app=FastAPI()

@app.get("/")
def greet():
    return {"msg": "Hello, RestAPI working"}


@app.post('/predict')
def predict_output(item:Data):
    item=item.dict()
    input_data=list(item.values())
    predicted_class=model.predict([input_data])

    if predicted_class[0] == 1:
        return {"msg":"Note is Fake"}
    else:
        return {"msg":"Note is Authentic"}









