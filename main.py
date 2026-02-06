from fastapi import FastAPI

app = FastAPI()

@app.get("/add")
def add(x, y):
    return {"result": (x + y)}