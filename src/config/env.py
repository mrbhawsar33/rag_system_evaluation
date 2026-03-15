from dotenv import load_dotenv
import os
# from huggingface_hub import login

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
# login(token=os.getenv("HF_TOKEN"))

# expose HF token to huggingface hub
if HF_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN