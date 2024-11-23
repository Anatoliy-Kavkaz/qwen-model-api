from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Загрузка модели
model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Создание FastAPI-приложения
app = FastAPI()

# Модель запроса
class Request(BaseModel):
    prompt: str
    max_tokens: int = 128

@app.post("/generate/")
async def generate_text(request: Request):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=request.max_tokens)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
