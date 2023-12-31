
import configparser
import erniebot
import gradio as gr
import warnings

import torch

warnings.filterwarnings("ignore")

def download_config():
    import shutil
    torch.hub.download_url_to_file(
        'https://github.com/isLinXu/issues/files/13178289/config.ini.zip',
        'config.ini.zip')
    shutil.unpack_archive('config.ini.zip', './', 'zip')

download_config()

# 从配置文件中读取 access_token
config = configparser.ConfigParser()
config.read("config.ini")
access_token = config.get("erniebot", "access_token")

# 设置认证参数
erniebot.api_type = "aistudio"
erniebot.access_token = access_token

# 定义与 Ernie 机器人的对话函数
def chat_with_erniebot(model, user_input):
    response = erniebot.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": user_input}]
    )
    return response.get_result()


# 设置 Gradio 界面
iface = gr.Interface(
    fn=chat_with_erniebot,
    inputs=[
        gr.inputs.Dropdown(
            choices=[
                "ernie-bot",
                "ernie-bot-turbo",
                "ernie-bot-4",
                "ernie-text-embedding",
                "ernie-vilg-v2",
            ],
            label="选择模型",
        ),
        gr.inputs.Textbox(lines=5, label="输入你想问 Ernie 机器人的问题"),
    ],
    outputs=gr.outputs.Textbox(type='text', label="Ernie 机器人的回答"),
    title="与 Ernie 机器人聊天",
    description="选择一个模型，然后输入你想问 Ernie 机器人的问题，点击提交按钮获取回答。",
)


# 启动 Gradio 界面
iface.launch()
