from fastapi import  FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.router import ocr_router

app = FastAPI()

app.include_router(ocr_router.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}