import io
from datetime import datetime
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

import PIL.ImageOps
import torch
import torchvision.transforms as transforms
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from transformers import pipeline

from models.model import MnistNNForNeurologDemo

app: FastAPI = FastAPI(
    title="Neurolog demo backend.",
    description="Backend for the Neurolog demo.",
    version="0.0.1",
    terms_of_service="",
    contact={
        "name": "Neurolog",
    },
    # license_info={
    #     "name": "MIT",
    #     "url": "https://choosealicense.com/licenses/mit/",
    # },
)

origins: List[str] = ["https://msthoma.github.io"]
# origins: List[str] = ["*"]

# Configure the CORS policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

network = MnistNNForNeurologDemo()
# network.load_state_dict(
#     torch.load(Path.cwd().parent / "models" / "nn_for_neurolog_demo_epoch_100.mdl")
# )
network.load_state_dict(
    torch.load(
        "./models/nn_for_neurolog_demo_epoch_080_narrow.mdl",
        map_location=torch.device("cpu"),
    )
)
img_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
)
pipe = pipeline(
    task="image-classification",
    model="farleyknight-org-username/vit-base-mnist",
)


@app.post("/deduce")
async def deduce(files: List[UploadFile] = File(...)):
    filenames = [f.filename for f in files]

    contents = [await f.read() for f in files]

    imgs = [
        PIL.ImageOps.invert(Image.open(io.BytesIO(i)).resize((28, 28)).convert("L"))
        for i in contents
    ]

    for img, filename in zip(imgs, filenames):
        img.save(Path.cwd() / filename)

    # stack photos and infer with network
    preds = network(torch.stack([img_transform(img) for img in imgs], dim=0))

    # softmax results to get classifications (NN does not have a softmax after final layer)
    predicted_digits = (
        torch.argmax(torch.nn.functional.softmax(preds, dim=1), dim=1).numpy().tolist()
    )

    response = {filename: digit for filename, digit in zip(filenames, predicted_digits)}

    print(
        "Predictions at",
        datetime.now(ZoneInfo("Europe/Nicosia")).strftime("%y-%m-%d %H:%M:%S"),
        "-",
        response,
    )

    return JSONResponse(content=response, headers={"Access-Control-Allow-Origin": "*"})


@app.post("/deduce_vit")
async def deduce_vit(files: List[UploadFile] = File(...)):
    filenames = [f.filename for f in files]

    contents = [await f.read() for f in files]

    imgs = [
        PIL.ImageOps.invert(Image.open(io.BytesIO(i)).resize((28, 28)).convert("L"))
        for i in contents
    ]

    for img, filename in zip(imgs, filenames):
        img.save(Path.cwd() / filename)

    # preds look like this:
    # [
    #     [{"score": 0.9960125684738159, "label": "2"}, ...],
    #     [{"score": 0.9960125684738159, "label": "2"}, ...],
    # ]
    preds = pipe(imgs, top_k=10)
    predicted_digits = [int(pred[0]["label"]) for pred in preds]

    response = {filename: digit for filename, digit in zip(filenames, predicted_digits)}

    print(
        "Predictions at",
        datetime.now(ZoneInfo("Europe/Nicosia")).strftime("%y-%m-%d %H:%M:%S"),
        "-",
        response,
    )

    return JSONResponse(content=response, headers={"Access-Control-Allow-Origin": "*"})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, access_log=True)
