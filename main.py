from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AddRequest(BaseModel):
    x: int
    y: int

@app.get("/add")
def add(x: int, y: int):
    return {"x": x, "y": y, "result": x + y}

@app.post("/add")
def add_post(req: AddRequest):
    return {"x": req.x, "y": req.y, "result": req.x + req.y}
