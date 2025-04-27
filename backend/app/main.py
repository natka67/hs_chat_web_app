from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import download, classify  # Make sure these modules exist

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (good for development; restrict later for prod!)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include your routers
app.include_router(download.router)
app.include_router(classify.router)
