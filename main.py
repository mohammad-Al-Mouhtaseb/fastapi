from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}


from transformers import AutoProcessor, MusicgenForConditionalGeneration
import requests, scipy, torch, threading
app = Flask(__name__)

model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)
sampling_rate = model.config.audio_encoder.sampling_rate


def gen_fun(request):
    inputs = processor(
        text=['sad'],
        padding=True,
        return_tensors="pt",
    )
    audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=1202)
    scipy.io.wavfile.write('s1', rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

@app.get("/")
async def root():
    t = threading.Thread(target=gen_fun, args=[request])
    t.setDaemon(True)
    t.start()
    return {"res":"sucsess"}
