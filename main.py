from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import vllm
from langchain_community.llms import VLLM

# 定义请求模型
class CompletionRequest(BaseModel):
    prompt: str

# 定义响应模型
class CompletionResponse(BaseModel):
    prompt: str
    completion: str

app = FastAPI()

# 读取配置文件
def load_config():
    with open('config.json', 'r') as config_file:
        return json.load(config_file)

config = load_config()

# 初始化VLLM模型
llm = VLLM(
    model=config['vllm']['model'],
    max_new_tokens=config['vllm']['max_new_tokens'],
    top_k=config['vllm']['top_k'],
    top_p=config['vllm']['top_p'],
    temperature=config['vllm']['temperature']
)

# 定义生成文本的路由
@app.post("/generate", response_model=CompletionResponse)
async def generate_text(request: CompletionRequest):
    try:
        # 使用VLLM生成文本
        completion = llm(request.prompt)
        return CompletionResponse(prompt=request.prompt, completion=completion)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 运行FastAPI服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config['server']['host'], port=config['server']['port'])

@app.get("/models")
async def get_models():
    # 返回可用模型列表
    return {"models": [config['vllm']['model']]}

