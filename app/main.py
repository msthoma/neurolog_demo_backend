import io
from pathlib import Path
from typing import List

import PIL.ImageOps
import torch
import torchvision.transforms as transforms
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from models.model import Sum2NN

description: str = """
Backend for the Neurolog demo.
"""

# FastAPI for
app: FastAPI = FastAPI(
    title="Neurolog demo backend.",
    description=description,
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

origins: List[str] = ["https://msthoma.github.io/"]

# Configure the CORS policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/deduce")
async def deduce(files: List[UploadFile] = File(...)):
    network = Sum2NN()
    network.load_state_dict(torch.load(Path.cwd().parent / "models" / "model_samples_3000_iter_8700_epoch_3.mdl"))
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    filenames = [f.filename for f in files]

    contents = [await f.read() for f in files]
    imgs = [PIL.ImageOps.invert(Image.open(io.BytesIO(i)).resize((28, 28)).convert("L")) for i in contents]
    for img, filename in zip(imgs, filenames):
        img.save(Path.cwd() / filename)

    preds = [network(img_transform(img).unsqueeze(0)) for img in imgs]
    print(preds)
    preds = {filename: torch.argmax(pred).item() for filename, pred in zip(filenames, preds)}
    return JSONResponse(content=preds)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, access_log=True)
