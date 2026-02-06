from fastapi import FastAPI

app = FastAPI()

@app.get("/add")
def add(x: int, y: int):
    return {"x": x, "y": y, "result": x + y}
