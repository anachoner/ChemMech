import sys
from pathlib import Path

# Ensure backend directory is in Python path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import base64
from io import BytesIO

from model.reaction_predictor import ReactionPredictor
from utils.molecule_utils import smiles_to_image, validate_smiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ChemMech API",
    description="AI-Powered Chemical Reaction Mechanism Generator",
    version="1.0.0"
)

# CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model (global variable)
predictor = None


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    reactants: str
    model_type: str = "chemberta"
    num_beams: int = 5
    max_length: int = 512
    
    class Config:
        json_schema_extra = {
            "example": {
                "reactants": "CC(=O)O.CCO",
                "model_type": "chemberta",
                "num_beams": 5,
                "max_length": 512
            }
        }


class MechanismStep(BaseModel):
    """Model for a single mechanism step"""
    step: int
    description: str
    smiles: str
    image: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    success: bool
    product_smiles: str
    product_image: Optional[str] = None
    confidence: float
    mechanism_steps: List[MechanismStep]
    model_used: str
    error: Optional[str] = None


class ValidationResponse(BaseModel):
    """Response model for SMILES validation"""
    valid: bool
    smiles: str


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global predictor
    logger.info("🚀 Starting ChemMech API...")
    logger.info("📦 Loading reaction prediction model...")
    try:
        predictor = ReactionPredictor(model_type="chemberta")
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        predictor = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down ChemMech API...")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """
    Root endpoint - basic health check
    
    Returns:
        Basic API status and model loading state
    """
    return {
        "status": "online",
        "model_loaded": predictor is not None
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Detailed health check endpoint
    
    Returns:
        Detailed health status including model state
    """
    return {
        "status": "healthy" if predictor else "degraded",
        "model_loaded": predictor is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_reaction(request: PredictionRequest):
    """
    Predict reaction products and mechanism from reactants
    
    Args:
        request: PredictionRequest containing:
            - reactants: SMILES string (e.g., "CC(=O)O.CCO")
            - model_type: Type of model to use (default: "chemberta")
            - num_beams: Number of beams for beam search (default: 5)
            - max_length: Maximum sequence length (default: 512)
    
    Returns:
        PredictionResponse containing:
            - product_smiles: Predicted product SMILES
            - product_image: Base64 encoded product molecule image
            - confidence: Prediction confidence score (0-1)
            - mechanism_steps: List of reaction mechanism steps with images
            - model_used: Name of model used
    
    Raises:
        HTTPException 503: If model is not loaded
        HTTPException 400: If invalid SMILES provided
        HTTPException 500: If prediction fails
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Validate input SMILES
        if not validate_smiles(request.reactants):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid SMILES string provided: {request.reactants}"
            )
        
        logger.info(f"🔬 Predicting reaction for: {request.reactants}")
        
        # Run prediction
        result = predictor.predict(
            reactants=request.reactants,
            num_beams=request.num_beams,
            max_length=request.max_length
        )
        
        # Generate product image
        product_image = None
        if result["product_smiles"]:
            img = smiles_to_image(result["product_smiles"])
            if img:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                product_image = base64.b64encode(buffered.getvalue()).decode()
        
        # Generate mechanism step images
        mechanism_steps = []
        for i, step in enumerate(result["mechanism_steps"]):
            step_image = None
            if step["smiles"]:
                img = smiles_to_image(step["smiles"])
                if img:
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    step_image = base64.b64encode(buffered.getvalue()).decode()
            
            mechanism_steps.append(
                MechanismStep(
                    step=i + 1,
                    description=step["description"],
                    smiles=step["smiles"],
                    image=step_image
                )
            )
        
        logger.info(f"✅ Prediction complete: {result['product_smiles']} (confidence: {result['confidence']})")
        
        return PredictionResponse(
            success=True,
            product_smiles=result["product_smiles"],
            product_image=product_image,
            confidence=result["confidence"],
            mechanism_steps=mechanism_steps,
            model_used=request.model_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/validate-smiles", response_model=ValidationResponse)
async def validate_smiles_endpoint(smiles: str):
    """
    Validate a SMILES string
    
    Args:
        smiles: SMILES string to validate
    
    Returns:
        ValidationResponse with validation result
    """
    is_valid = validate_smiles(smiles)
    return ValidationResponse(
        valid=is_valid,
        smiles=smiles
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)