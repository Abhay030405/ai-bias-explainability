"""
Main FastAPI application for FairLens AI
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings
from app.utils.logger import get_logger
from app.api.routes_explain import router as explain_router
from app.api.routes_bias import router as bias_router  # ⭐ ADD THIS LINE

logger = get_logger()

# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="""
    ## FairLens AI - Making ML Models Transparent and Fair
    
    ### Features:
    * 🔍 **Explainability**: SHAP and LIME-based feature importance
    * ⚖️ **Bias Detection**: Demographic parity, equal opportunity, disparate impact
    * 🤖 **LLM Summaries**: Natural language explanations and recommendations
    * 📊 **Interactive Dashboard**: Streamlit-based UI for visualization
    
    ### Endpoints:
    * `/api/explain` - Get SHAP and LIME explanations
    * `/api/bias` - Compute fairness metrics (JSON input)
    * `/api/bias/upload` - Compute fairness metrics (CSV upload)
    * `/api/bias/thresholds` - Get fairness thresholds
    * `/api/summary` - Generate LLM summaries (Phase 3)
    * `/health` - Health check
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(explain_router)
app.include_router(bias_router)  # ⭐ ADD THIS LINE

@app.on_event("startup")
async def startup_event():
    """Actions to perform on application startup"""
    logger.info("🚀 Starting FairLens AI API")
    logger.info(f"📝 API Documentation: http://{settings.api_host}:{settings.api_port}/docs")
    
    # Pre-load model to catch any issues early
    try:
        from app.dependencies import get_model, get_feature_names
        model = get_model()
        features = get_feature_names()
        logger.success(f"✅ Model loaded successfully with {len(features)} features")
    except Exception as e:
        logger.error(f"❌ Failed to pre-load model: {e}")
        logger.warning("⚠️  API will start but explainability endpoints may fail")


@app.on_event("shutdown")
async def shutdown_event():
    """Actions to perform on application shutdown"""
    logger.info("🛑 Shutting down FairLens AI API")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "service": "FairLens AI",
        "version": settings.api_version,
        "status": "running",
        "message": "Welcome to FairLens AI - Explainability and Bias Detection API",
        "documentation": "/docs",
        "endpoints": {
            "explainability": "/api/explain",
            "features": "/api/explain/features",
            "bias_json": "/api/bias",
            "bias_csv": "/api/bias/upload",
            "bias_thresholds": "/api/bias/thresholds",
            "health": "/health"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns service status and model availability
    """
    try:
        # Check if model is loadable
        from app.dependencies import get_model
        model = get_model()
        model_status = "healthy"
    except Exception as e:
        logger.error(f"Health check - model issue: {e}")
        model_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy" if model_status == "healthy" else "degraded",
        "service": "FairLens AI",
        "version": settings.api_version,
        "model_status": model_status,
        "components": {
            "api": "healthy",
            "model": model_status,
            "logging": "healthy"
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )