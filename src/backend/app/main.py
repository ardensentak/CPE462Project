from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import predict  # Import your predict router

app = FastAPI(
    title="Image Classification API",
    description="API to classify uploaded images using a trained ML model.",
    version="1.0.0"
)

# Allow CORS (important for frontend to connect during dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(predict.router, prefix="/api")

# Optional: health check
@app.get("/")
def read_root():
    return {"status": "API is running"}
