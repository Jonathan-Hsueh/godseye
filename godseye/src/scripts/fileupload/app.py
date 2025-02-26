from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from io import BytesIO
from PIL import Image

app = FastAPI()

model = YOLO("trainedmodel.pt")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your Vercel frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        results = model(image)
        annotated_image = results[0].plot()

        output_image = Image.fromarray(annotated_image)
        buffer = BytesIO()
        output_image.save(buffer, format="PNG")
        buffer.seek(0)

        return Response(content=buffer.getvalue(), media_type="image/png")
    except Exception as e:
        return Response(content=str(e), status_code=500)
