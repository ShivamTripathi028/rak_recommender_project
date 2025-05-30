# shivamtripathi028-rak_recommender_project/api_main.py
import logging
import os
import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json

# Add recommender_system to Python path
# This assumes api_main.py is in the root of RAK_RECOMMENDER_PROJECT
# and recommender_system is a direct subdirectory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from recommender_system.data_loader import DataLoader
from recommender_system.recommender import ProductRecommender
from recommender_system import config as sys_config # Import system's config

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RecommenderAPI")

# --- Initialize FastAPI App ---
app = FastAPI(title="RAK Product Recommender API")

# --- CORS Middleware ---
origins = [
    "http://localhost:8888",    # Default Netlify Dev port for the overall site proxy
    "http://localhost:8080",    # Default Vite dev server port for Project 1
    "http://localhost:5173",    # Common alternative Vite port
    "http://127.0.0.1:8888",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5173",
    "https://shadow-talk-to-rak.netlify.app",
    # Add your deployed Netlify frontend URL here later like:
    # "https://your-unified-app-name.netlify.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Or ["*"] for testing, then restrict
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"], # Allow OPTIONS for preflight requests
    allow_headers=["*"],
)

# --- Global Initializations for Recommender ---
# These variables will store the initialized instances or error messages.
data_loader_instance = None
recommender_instance = None
initialization_error_message = None

@app.on_event("startup")
async def startup_event():
    global data_loader_instance, recommender_instance, initialization_error_message
    logger.info("FastAPI application startup: Initializing recommender...")
    try:
        # base_dir is the directory where api_main.py is located
        # e.g., C:\Users\Sibi\Desktop\rak_recommender_project
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the recommender_system package directory
        recommender_system_dir = os.path.join(base_dir, 'recommender_system')

        # Construct full paths to data files using paths from config.py
        # sys_config.PRODUCT_FILE is 'data/product_table.csv' (relative to recommender_system_dir)
        product_file_path = os.path.join(recommender_system_dir, sys_config.PRODUCT_FILE)
        feature_file_path = os.path.join(recommender_system_dir, sys_config.FEATURE_FILE)
        mapping_file_path = os.path.join(recommender_system_dir, sys_config.MAPPING_FILE)

        logger.info(f"Attempting to load product data from: {product_file_path}")
        if not os.path.exists(product_file_path):
            raise FileNotFoundError(f"Product file not found at: {product_file_path}")
        if not os.path.exists(feature_file_path):
            raise FileNotFoundError(f"Feature file not found at: {feature_file_path}")
        if not os.path.exists(mapping_file_path):
            raise FileNotFoundError(f"Mapping file not found at: {mapping_file_path}")

        data_loader_instance = DataLoader(
            product_file=product_file_path,
            feature_file=feature_file_path,
            mapping_file=mapping_file_path
        )

        if not data_loader_instance.load_data():
            initialization_error_message = "Failed to load data via DataLoader."
        elif not data_loader_instance.preprocess_data():
            initialization_error_message = "Failed to preprocess data via DataLoader."
        else:
            products_df, features_df, product_feature_map_df = data_loader_instance.get_data()
            if products_df is None or products_df.empty:
                initialization_error_message = "Product data is empty after loading/preprocessing."
            else:
                recommender_instance = ProductRecommender(products_df, features_df, product_feature_map_df)
                logger.info("Recommender initialized successfully for FastAPI app.")
        
        if initialization_error_message:
            logger.error(f"Recommender Initialization Error on Startup: {initialization_error_message}")

    except Exception as e:
        initialization_error_message = f"Critical error during recommender initialization: {str(e)}"
        logger.error(initialization_error_message, exc_info=True)

# Pydantic model for request body validation
class ClientRequirements(BaseModel):
    clientInfo: dict | None = None # Making these optional as per example JSON
    region: dict | None = None
    deployment: dict | None = None
    application: dict | None = None
    scale: str | None = None
    connectivity: dict | None = None
    power: list[str] | None = None
    additionalDetails: str | None = None
    # Add any other fields your frontend might send, ensuring types match

@app.post("/recommend")
async def get_recommendations_api(requirements: ClientRequirements):
    # Global variables are read here; no 'global' keyword needed for reading.
    if initialization_error_message:
        logger.error(f"API call failed due to initialization error: {initialization_error_message}")
        raise HTTPException(status_code=503, detail=f"Recommender service not ready: {initialization_error_message}")
    if not recommender_instance:
        logger.error("API call failed because recommender_instance is None.")
        raise HTTPException(status_code=503, detail="Recommender service is unavailable.")

    try:
        # Convert Pydantic model to dict. exclude_unset=True means fields not provided in JSON won't be in dict
        client_requirements_json = requirements.model_dump(exclude_unset=True) 
        
        if not client_requirements_json: # Check if the resulting dict is empty
             logger.warning("Received empty requirements after Pydantic parsing.")
             raise HTTPException(status_code=400, detail="Empty or invalid requirements payload.")

        logger.info(f"Received requirements for /recommend: {json.dumps(client_requirements_json, indent=2)}")
        
        recommendations = recommender_instance.recommend(client_requirements_json, top_n=3)
        logger.info(f"Generated {len(recommendations)} recommendations.")
        
        return recommendations
    except HTTPException as e: # Re-raise HTTPExceptions from parsing or validation if any
        raise e
    except Exception as e:
        logger.error(f"Error processing /recommend request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    if initialization_error_message or not recommender_instance:
        return {"status": "unhealthy", "reason": initialization_error_message or "Recommender not initialized"}
    return {"status": "healthy"}

# To run locally (for testing Project 2 API standalone):
# In your terminal, from RAK_RECOMMENDER_PROJECT root:
# uvicorn api_main:app --reload --port 8001
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001)) # Railway will set PORT env var. Default to 8001 for local.
    logger.info(f"Starting Uvicorn server directly on 0.0.0.0:{port} (for local testing via 'python api_main.py')...")
    uvicorn.run("api_main:app", host="0.0.0.0", port=port, reload=True)