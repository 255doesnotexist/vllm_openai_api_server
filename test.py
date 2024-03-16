import json
from vllm import LLM, SamplingParams

# 从config.json文件中加载模型配置
def load_config():
    with open('config.json', 'r') as config_file:
        return json.load(config_file)

# 初始化VLLM模型
def init_model(config):
    return LLM(
        model=config['vllm']['model'],
        dtype='float16'
    ),  SamplingParams(
        max_new_tokens=config['vllm']['max_new_tokens'],
        top_k=config['vllm']['top_k'],
        top_p=config['vllm']['top_p'],
        temperature=config['vllm']['temperature'])

# 主函数
def main():
    config = load_config()
    llm, sampling_params = init_model(config)
    
    print("VLLM模型已加载。请输入提示（输入'退出'以结束）：")
    
    while True:
        prompt = input("提示> ")
        if prompt.lower() == '退出':
            break
        completion = llm(prompt)
        print("生成的文本:", completion)

if __name__ == "__main__":
    main()
