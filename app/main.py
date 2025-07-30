from fastapi import FastAPI
from .routes import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Prompt Library API",
    description="API to fetch categorized prompt libraries from MongoDB",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
