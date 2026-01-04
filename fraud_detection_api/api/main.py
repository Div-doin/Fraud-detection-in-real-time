# fraud_detection_api/api/main.py
"""
Fraud Detection API (MongoDB + FastAPI, async)
- Loads a pre-trained CatBoost model (joblib bundle).
- Accepts preprocessed/scaled features (as in training).
- Logs predictions and alerts to MongoDB (async motor).
- Sends EMAIL alerts for HIGH/CRITICAL fraud predictions.
"""

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Header,
    WebSocket,
    WebSocketDisconnect,
    APIRouter,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import os
import time
import logging
import asyncio
import json
from pymongo import MongoClient  # not used directly, but ok to keep
from concurrent.futures import ThreadPoolExecutor

# toggle via env var: set TEST_ALLOW_EXTREME_ALERTS=1 to force alerts even for extreme probs
TEST_ALLOW_EXTREME_ALERTS = os.getenv("TEST_ALLOW_EXTREME_ALERTS", "0") == "1"

# Optional: Azure Application Insights log exporter (non-blocking if not installed)
try:
    from opencensus.ext.azure.log_exporter import AzureLogHandler
    AZURE_INSIGHTS_AVAILABLE = True
except Exception:
    AZURE_INSIGHTS_AVAILABLE = False

# Async MongoDB utilities (must be provided in api/database.py)
# Expected functions/variables from api/database.py:
#   connect_to_mongo(), close_mongo_connection(), get_db() (dependency),
#   get_mongo_uri_info(), MONGO_PREDICTION_COLLECTION, MONGO_ALERTS_COLLECTION
try:
    from fraud_detection_api.api.database import (
        connect_to_mongo,
        close_mongo_connection,
        get_db as get_mongo_db,
        get_mongo_uri_info,
        MONGO_PREDICTION_COLLECTION,
        MONGO_ALERTS_COLLECTION,
    )
    MONGO_AVAILABLE = True
except Exception as e:
    get_mongo_db = None
    connect_to_mongo = None
    close_mongo_connection = None
    get_mongo_uri_info = None
    MONGO_PREDICTION_COLLECTION = "predictions"
    MONGO_ALERTS_COLLECTION = "alerts"
    MONGO_AVAILABLE = True

router = APIRouter()
# ---------------------------------------------------------------------
# WebSocket Connection Manager (for real-time alerts)
# ---------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        """Send message to all connected clients."""
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                # remove dead/broken connection
                self.disconnect(connection)


manager = ConnectionManager()

# threadpool for blocking model calls
executor = ThreadPoolExecutor(max_workers=2)

# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "api.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("api.main")

# Attach Azure App Insights handler if configured
if AZURE_INSIGHTS_AVAILABLE and os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    try:
        handler = AzureLogHandler(
            connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        )
        logger.addHandler(handler)
        logger.info("Azure Application Insights handler attached")
    except Exception as e:
        logger.warning(f"Could not attach AzureLogHandler: {e}")
# ---------------------------------------------------------------------
# Config / Model loading
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = os.getenv("MODEL_PATH", str(MODEL_DIR / "Variant III_CatBoost.pkl"))
STATIC_DIR = BASE_DIR / "static"
API_KEY = os.getenv("API_KEY", None)
MONGO_URI_INFO = get_mongo_uri_info() if get_mongo_uri_info else None

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection using CatBoost model (preprocessed features)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ✅ CORS (helps if you open HTML from file:// or different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Serve static files (HTML/JS/CSS) from ./static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Email alert config (CHANGE THESE OR USE ENV VARS)
ALERT_FROM_EMAIL = os.getenv(
    "ALERT_FROM_EMAIL", "sahaydivyeshmukesh23csds@rnsit.ac.in"
)
ALERT_TO_EMAIL = os.getenv(
    "ALERT_TO_EMAIL", "divyeshsahay048@gmail.com"
)
ALERT_EMAIL_PASSWORD = os.getenv(
    "ALERT_EMAIL_PASSWORD", "ysiy zryc sgmo iwcq"
)

# Replace lines ~200-235 in main.py with this:

model = None
threshold: float = 0.5
expected_features: List[str] = []
scaler = None  # ✅ ADD THIS
scaler_feature_names = None  # ✅ ADD THIS

# Load model
try:
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        logger.error(f"Model file not found at: {MODEL_PATH}")
    else:
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict):
            model = bundle.get("model")
            threshold = float(bundle.get("threshold", 0.15))
        else:
            model = bundle
            threshold = float(getattr(bundle, "threshold", 0.15))

        if model is not None and hasattr(model, "feature_names_"):
            try:
                expected_features = list(model.feature_names_) or []
            except Exception:
                expected_features = []
        else:
            # Some sklearn Pipelines expose feature_names_in_
            try:
                expected_features = list(getattr(model, "feature_names_in_", [])) or []
            except Exception:
                expected_features = []

        logger.info(f"Model loaded: {MODEL_PATH}")
        logger.info(
            f"Threshold: {threshold}, expected_features_count: {len(expected_features)}"
        )
except Exception as e:
    logger.exception("Failed to load model")
    model = None
    threshold = 0.15
    expected_features = []

# ✅ Load scaler AFTER model
SCALER_PATH = MODEL_DIR / "scaler.pkl"
try:
    if SCALER_PATH.exists():
        logger.info(f"Loading scaler from {SCALER_PATH}")
        scaler_data = joblib.load(SCALER_PATH)
        scaler = scaler_data['scaler']
        scaler_feature_names = scaler_data['feature_names']
        logger.info(f"✅ Loaded StandardScaler successfully")
        logger.info(f"   Scaler features: {len(scaler_feature_names)}")
    else:
        logger.warning(f"⚠️ Scaler not found at {SCALER_PATH}")
        logger.warning("⚠️ Predictions will be INCORRECT without scaler!")
        logger.warning("⚠️ Run: python create_scaler.py")
        scaler = None
        scaler_feature_names = None
except Exception as e:
    logger.exception(f"Failed to load scaler: {e}")
    scaler = None
    scaler_feature_names = None

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def verify_api_key(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    """Verify API key if configured (synchronous)."""
    if API_KEY and x_api_key != API_KEY:
        logger.warning("Invalid API key attempt")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key


async def get_db_dependency():
    """Yield async Motor DB object or None (if Mongo not available)."""
    if MONGO_AVAILABLE and get_mongo_db:
        async for db in get_mongo_db():
            yield db
            return
    else:
        yield None


def determine_risk_level(prob: float) -> str:
    if prob < 0.15:
        return "LOW"
    if prob < 0.4:
        return "MEDIUM"
    if prob < 0.7:
        return "HIGH"
    return "CRITICAL"


# ---------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Fraud Detection API...")
    if MONGO_AVAILABLE and connect_to_mongo:
        try:
            await connect_to_mongo()
            logger.info("Connected to MongoDB")
        except Exception as e:
            logger.warning(f"Could not connect to MongoDB on startup: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Fraud Detection API...")
    if MONGO_AVAILABLE and close_mongo_connection:
        try:
            await close_mongo_connection()
            logger.info("MongoDB connection closed")
        except Exception as e:
            logger.warning(f"Error closing MongoDB connection: {e}")


# ---------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------
# Replace the Transaction class in main.py (around line 280)

class Transaction(BaseModel):
    """
    Transaction schema - all features are RAW values that will be scaled by StandardScaler
    """

    model_config = ConfigDict(json_schema_extra={"example": {}})

    income: float
    name_email_similarity: float
    prev_address_months_count: float
    current_address_months_count: float
    customer_age: float
    days_since_request: float
    intended_balcon_amount: float
    payment_type: float
    zip_count_4w: float
    velocity_6h: float
    velocity_24h: float
    velocity_4w: float
    bank_branch_count_8w: float
    date_of_birth_distinct_emails_4w: float
    employment_status: float
    credit_risk_score: float
    email_is_free: float
    housing_status: float
    phone_home_valid: float
    phone_mobile_valid: float
    bank_months_count: float
    has_other_cards: float
    proposed_credit_limit: float
    foreign_request: float
    source: float
    session_length_in_minutes: float
    device_os: float
    keep_alive_session: float
    device_distinct_emails_8w: float
    device_fraud_count: float
    month: float
    x1: float = 0.0  # ✅ ADD THIS - dummy feature
    x2: float = 0.0


class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    threshold_used: float
    timestamp: str
    model_version: str
    processing_time_ms: float
    alert_triggered: bool
    alert_message: Optional[str] = None


class BatchTransaction(BaseModel):
    transactions: List[Transaction]


# ---------------------------------------------------------------------
# Debug / Health / Info endpoints
# ---------------------------------------------------------------------
@app.get("/debug/model_features")
async def debug_model_features():
    return {
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "expected_features_count": len(expected_features),
        "expected_features_sample": expected_features[:20]
        if expected_features
        else "Not available",
        "mongo_available": MONGO_AVAILABLE,
        "mongo_uri_info": MONGO_URI_INFO,
    }


@app.get("/")
async def root():
    return {
        "service": "Fraud Detection API",
        "status": "online",
        "model_loaded": model is not None,
        "expected_features_count": len(expected_features),
        "mongo_available": MONGO_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
        "threshold": float(threshold),
        "expected_features_count": len(expected_features),
        "mongo_available": MONGO_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/model/info")
async def model_info(api_key: str = Depends(verify_api_key)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(model).__name__,
        "model_version": "1.0",
        "training_dataset": "Variant III",
        "threshold": float(threshold),
        "features_count": len(expected_features),
        "expected_features": expected_features
        if expected_features
        else "Not available",
        "accepts_preprocessed_data": True,
        "model_path": str(MODEL_PATH),
    }


# ---------------------------------------------------------------------
# PREDICT (single) with Mongo logging + EMAIL alert + WebSocket broadcast
# ---------------------------------------------------------------------
# Replace the predict_fraud function in main.py with this fixed version

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(
    transaction: Transaction,
    db=Depends(get_db_dependency),
    api_key: str = Depends(verify_api_key),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # 1) Build DataFrame from input
        transaction_dict = transaction.model_dump()
        df = pd.DataFrame([transaction_dict])

        # 2) **FIX: Strict feature alignment**
        if not expected_features:
            logger.error("Model has no feature_names_ - cannot proceed safely")
            raise HTTPException(
                status_code=500, 
                detail="Model feature names not available - model may be corrupted"
            )

        # Check for missing features — auto-fill defaults rather than failing
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            # Fill missing features with zeros (or another default) so model receives full vector
            logger.warning(f"Missing model features detected, auto-filling with 0.0: {missing_features}")
            for f in missing_features:
                df[f] = 0.0  # <-- change default here if you need something else
        # Continue — features will be aligned below using expected_features order

        # **CRITICAL FIX: Use features in exact model order**
        df_for_pred = df[expected_features]
        
        # Log feature statistics for debugging
        logger.info(f"Input features BEFORE scaling - min: {df_for_pred.values.min():.4f}, max: {df_for_pred.values.max():.4f}, mean: {df_for_pred.values.mean():.4f}")

        # ✅ CRITICAL FIX: Apply StandardScaler transformation
        if scaler is not None:
            logger.info("✅ Applying StandardScaler transformation...")
            
            # Check if input looks already scaled (suspicious if all in 0-1)
            input_min = df_for_pred.values.min()
            input_max = df_for_pred.values.max()
            input_mean = df_for_pred.values.mean()
            
            if input_min >= 0 and input_max <= 1:
                logger.warning("⚠️  WARNING: Input appears to be in [0,1] range")
                logger.warning("⚠️  Your HTML form may be sending pre-scaled values")
                logger.warning("⚠️  Model expects RAW values that will be scaled by StandardScaler")
                
                # Try to detect if these are truly raw values that happen to be in 0-1
                # or if they're pre-normalized
                if input_mean < 0.3 or input_mean > 0.7:
                    # Likely raw binary features or actual 0-1 data
                    logger.info("✅ Data appears legitimate (mean not centered at 0.5)")
                else:
                    logger.error("❌ Data appears to be pre-normalized from sliders!")
                    logger.error("❌ This will produce WRONG predictions!")
                    logger.error("❌ Fix your HTML form to send RAW values!")
            
            # Apply scaling
            X_scaled = scaler.transform(df_for_pred)
            logger.info(f"✅ Scaled features - min: {X_scaled.min():.4f}, max: {X_scaled.max():.4f}, mean: {X_scaled.mean():.4f}")
            logger.info(f"First 5 scaled features: {X_scaled[0, :5]}")
            
            # Use scaled features for prediction
            X_for_prediction = X_scaled
        else:
            logger.error("❌ NO SCALER AVAILABLE - Using raw values (WILL BE WRONG!)")
            logger.error("❌ Run: python create_scaler.py")
            X_for_prediction = df_for_pred.values

        # 3) Predict probability - use the scaled dataframe
        try:
            fraud_probability = float(model.predict_proba(X_for_prediction)[0, 1])
            logger.info(f"Prediction probability: {fraud_probability:.6f}")
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
        # **FIX: Sanity check on probability**
        if fraud_probability < 0 or fraud_probability > 1:
            logger.error(f"Invalid probability {fraud_probability} - model error")
            raise HTTPException(
                status_code=500,
                detail="Model returned invalid probability"
            )

        is_fraud = bool(fraud_probability >= threshold)
        risk_level = determine_risk_level(fraud_probability)
        transaction_id = f"TXN_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
        processing_time = (time.time() - start_time) * 1000.0

        # **FIX: More conservative alert logic**
        # Only trigger alerts for genuinely high-risk transactions
        alert_triggered = False
        alert_message = None
        
        # Block extreme probabilities (model likely malfunctioning)
        if fraud_probability >= 0.999 or fraud_probability <= 0.001:
            logger.warning(
                f"Model returned extreme probability {fraud_probability:.6f} - "
                f"possible feature preprocessing issue. Features: {list(df_for_pred.iloc[0].values[:5])}"
            )
            if not TEST_ALLOW_EXTREME_ALERTS:
                alert_message = "Blocked: extreme model output (check feature preprocessing)"
                is_fraud = False  # Override fraud flag
                risk_level = "UNKNOWN"
        else:
            # Normal alert logic
            alert_triggered = is_fraud and risk_level in ("MEDIUM","HIGH", "CRITICAL")
            if alert_triggered:
                alert_message = (
                    f"⚠️ ALERT: High-risk transaction detected! "
                    f"ID={transaction_id}, Risk={risk_level}, Prob={fraud_probability:.3f}"
                )

        # Log more details for debugging
        logger.info(
            f"Prediction {transaction_id} | fraud={is_fraud} | "
            f"prob={fraud_probability:.4f} | risk={risk_level} | "
            f"threshold={threshold:.4f} | alert={alert_triggered} | "
            f"time={processing_time:.2f}ms"
        )

        # 4) Log prediction to MongoDB
        if MONGO_AVAILABLE and db is not None:
            try:
                log_doc = {
                    "transaction_id": transaction_id,
                    "payload": transaction_dict,
                    "fraud_probability": float(fraud_probability),
                    "is_fraud": is_fraud,
                    "risk_level": risk_level,
                    "threshold": float(threshold),
                    "processing_time_ms": processing_time,
                    "alert_triggered": alert_triggered,
                    "timestamp": datetime.utcnow(),
                }
                await db[MONGO_PREDICTION_COLLECTION].insert_one(log_doc)

                # Only log to alerts collection if genuinely triggered
                if alert_triggered:
                    alert_doc = {
                        "transaction_id": transaction_id,
                        "risk_level": risk_level,
                        "fraud_probability": float(fraud_probability),
                        "message": alert_message,
                        "timestamp": datetime.utcnow(),
                        "status": "NEW",
                    }
                    await db[MONGO_ALERTS_COLLECTION].insert_one(alert_doc)
            except Exception as e:
                logger.warning(f"Could not write to MongoDB: {e}")

        # 5) WebSocket broadcast (only for real alerts)
        if alert_triggered:
            try:
                ws_message = json.dumps({
                    "type": "new_alert",
                    "transaction_id": transaction_id,
                    "risk_level": risk_level,
                    "fraud_probability": fraud_probability,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                await manager.broadcast(ws_message)
                logger.info("📡 WebSocket alert broadcasted")
            except Exception as e:
                logger.warning(f"WebSocket broadcast failed: {e}")

        # 6) Send EMAIL alert (only for genuine HIGH/CRITICAL)
        if alert_triggered and ALERT_FROM_EMAIL and ALERT_TO_EMAIL and ALERT_EMAIL_PASSWORD:
            try:
                import smtplib
                from email.mime.text import MIMEText

                msg = MIMEText(
                    f"⚠️ FRAUD ALERT ⚠️\n\n"
                    f"Transaction ID: {transaction_id}\n"
                    f"Risk Level: {risk_level}\n"
                    f"Fraud Probability: {fraud_probability:.3f}\n"
                    f"Timestamp: {datetime.utcnow().isoformat()}\n"
                )
                msg["Subject"] = f"FRAUD ALERT - {risk_level}"
                msg["From"] = ALERT_FROM_EMAIL
                msg["To"] = ALERT_TO_EMAIL

                with smtplib.SMTP("smtp.gmail.com", 587) as server:
                    server.starttls()
                    server.login(ALERT_FROM_EMAIL, ALERT_EMAIL_PASSWORD)
                    server.send_message(msg)

                logger.info("📧 Email alert sent")
            except Exception as e:
                logger.warning(f"Email alert failed: {e}")

        # 7) Return response
        return PredictionResponse(
            transaction_id=transaction_id,
            is_fraud=is_fraud,
            fraud_probability=float(fraud_probability),
            risk_level=risk_level,
            threshold_used=float(threshold),
            timestamp=datetime.utcnow().isoformat(),
            model_version=type(model).__name__ if model else "unknown",
            processing_time_ms=processing_time,
            alert_triggered=alert_triggered,
            alert_message=alert_message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ---------------------------------------------------------------------
# BATCH PREDICT
# ---------------------------------------------------------------------
@app.post("/predict/batch")
async def predict_batch(
    batch: BatchTransaction,
    db=Depends(get_db_dependency),
    api_key: str = Depends(verify_api_key),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        df = pd.DataFrame([t.model_dump() for t in batch.transactions])

        use_values = False
        if expected_features:
            available_features = [f for f in expected_features if f in df.columns]
            if len(available_features) == 0:
                use_values = True
            else:
                df_for_pred = df[available_features]
        else:
            use_values = True

        if use_values:
            X_input = df.values
        else:
            X_input = df_for_pred

        try:
            if (
                not use_values
                and hasattr(model, "feature_names_")
                and model.feature_names_
            ):
                probabilities = model.predict_proba(X_input)[:, 1]
            else:
                arr = X_input if isinstance(X_input, np.ndarray) else X_input.values
                probabilities = model.predict_proba(arr)[:, 1]
        except Exception as e:
            logger.exception("Batch prediction failed")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

        predictions = (probabilities >= threshold).astype(bool)

        results = []
        docs_to_insert = []
        for idx, (prob, pred) in enumerate(zip(probabilities, predictions)):
            risk = determine_risk_level(float(prob))
            results.append(
                {
                    "transaction_index": idx,
                    "is_fraud": bool(pred),
                    "fraud_probability": float(prob),
                    "risk_level": risk,
                }
            )
            if MONGO_AVAILABLE and db is not None:
                docs_to_insert.append(
                    {
                        "transaction_id": f"TXN_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{idx}",
                        "payload": batch.transactions[idx].model_dump(),
                        "fraud_probability": float(prob),
                        "is_fraud": bool(pred),
                        "risk_level": risk,
                        "threshold": float(threshold),
                        "processing_time_ms": None,
                        "timestamp": datetime.utcnow(),
                    }
                )

        if MONGO_AVAILABLE and db is not None and docs_to_insert:
            try:
                asyncio.create_task(
                    db[MONGO_PREDICTION_COLLECTION].insert_many(docs_to_insert)
                )
            except Exception as e:
                logger.warning(f"Failed to insert batch logs to MongoDB: {e}")

        processing_time = (time.time() - start_time) * 1000.0
        logger.info(
            f"Batch processed: size={len(results)} frauds={sum(r['is_fraud'] for r in results)} time_ms={processing_time:.2f}"
        )

        return {
            "total_transactions": len(results),
            "fraudulent_count": sum(r["is_fraud"] for r in results),
            "processing_time_ms": processing_time,
            "predictions": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected batch error")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


# ---------------------------------------------------------------------
# WebSocket endpoint for real-time alerts
# ---------------------------------------------------------------------
@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Keep the connection open; we don't expect messages from client
        while True:
            await websocket.receive_text()  # ignore content, just keep alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# ---------------------------------------------------------------------
# Alerts API (for alerts dashboard)
# ---------------------------------------------------------------------
@app.get("/alerts")
async def get_alerts(
    limit: int = 50,
    db=Depends(get_db_dependency),
):
    """
    Return recent fraud alerts based on prediction logs.
    Only returns entries where alert_triggered = True.
    """
    if not MONGO_AVAILABLE or db is None:
        return []

    coll = db[MONGO_PREDICTION_COLLECTION]

    cursor = coll.find(
        {"alert_triggered": True}
    ).sort("timestamp", -1).limit(limit)

    docs = await cursor.to_list(length=limit)

    alerts = []
    for doc in docs:
        alerts.append(
            {
                "id": str(doc.get("_id")),
                "transaction_id": doc.get("transaction_id"),
                "amount": doc.get("payload", {}).get("proposed_credit_limit")
                          or doc.get("payload", {}).get("intended_balcon_amount"),
                "score": doc.get("fraud_probability"),
                "risk_level": doc.get("risk_level"),
                "prediction": "fraud" if doc.get("is_fraud") else "normal",
                "timestamp": doc.get("timestamp").isoformat()
                if doc.get("timestamp") else None,
            }
        )

    return alerts


# ---------------------------------------------------------------------
# Analytics endpoints
# ---------------------------------------------------------------------
@app.get("/analytics/summary")
async def analytics_summary(
    db=Depends(get_db_dependency),
    api_key: str = Depends(verify_api_key),
):
    if not MONGO_AVAILABLE or db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        coll = db[MONGO_PREDICTION_COLLECTION]
        total = await coll.count_documents({})
        fraud_count = await coll.count_documents({"is_fraud": True})

        avg_prob_cursor = coll.aggregate(
            [{"$group": {"_id": None, "avg_prob": {"$avg": "$fraud_probability"}}}]
        )
        avg_prob_res = await avg_prob_cursor.to_list(length=1)
        avg_prob = avg_prob_res[0]["avg_prob"] if avg_prob_res else 0.0

        avg_time_cursor = coll.aggregate(
            [
                {"$match": {"processing_time_ms": {"$ne": None}}},
                {"$group": {"_id": None, "avg_time": {"$avg": "$processing_time_ms"}}},
            ]
        )
        avg_time_res = await avg_time_cursor.to_list(length=1)
        avg_time = avg_time_res[0]["avg_time"] if avg_time_res else 0.0

        risk_cursor = coll.aggregate(
            [{"$group": {"_id": "$risk_level", "count": {"$sum": 1}}}]
        )
        risk_list = await risk_cursor.to_list(length=100)
        risk_distribution = {r["_id"]: r["count"] for r in risk_list}

        return {
            "total_transactions": total,
            "fraud_count": fraud_count,
            "fraud_rate": (fraud_count / total * 100) if total > 0 else 0,
            "avg_fraud_probability": float(avg_prob),
            "avg_response_time_ms": float(avg_time),
            "risk_distribution": risk_distribution,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.exception("Analytics summary failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/recent")
async def analytics_recent(
    limit: int = 50,
    db=Depends(get_db_dependency),
    api_key: str = Depends(verify_api_key),
):
    if not MONGO_AVAILABLE or db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        coll = db[MONGO_PREDICTION_COLLECTION]
        cursor = coll.find().sort("timestamp", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        results = [
            {
                "transaction_id": doc.get("transaction_id"),
                "timestamp": doc.get("timestamp").isoformat()
                if doc.get("timestamp")
                else None,
                "is_fraud": doc.get("is_fraud"),
                "fraud_probability": doc.get("fraud_probability"),
                "risk_level": doc.get("risk_level"),
                "processing_time_ms": doc.get("processing_time_ms"),
            }
            for doc in docs
        ]

        return {"count": len(results), "transactions": results}
    except Exception as e:
        logger.exception("Analytics recent failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------
# Verbose debug predict endpoint (replacement)
# ---------------------------------------------------------------------
@router.post("/debug_predict")
@router.post("/verbose_debug_predict")
async def verbose_debug_predict(transaction: Transaction):
    """
    Robust verbose debug endpoint:
    - Builds feature vector using expected_features if available (model.feature_names_)
    - Falls back to using the transaction dict values in insertion order
    - Calls model.predict_proba() and model.predict() on the input and a set of test vectors
    - Returns model_feature_names (if present), fv, fv_len, and test results
    """
    # Use the global model already loaded in module
    global model, expected_features, executor

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get the raw transaction dict (pydantic)
    tx_dict = transaction.model_dump()
    # Build a single-row DataFrame so column ordering is explicit
    df = pd.DataFrame([tx_dict])

    # Determine the feature order to use
    if expected_features:
        # use only expected features that exist in incoming payload
        available_features = [f for f in expected_features if f in df.columns]
        if not available_features:
            # none of the expected features present — fallback to all columns provided
            features_order = list(df.columns)
            df_for_pred = df[features_order]
        else:
            features_order = available_features
            df_for_pred = df[features_order]
    else:
        # If model did not expose feature_names_, try to get them from the model object
        if hasattr(model, "feature_names_in_"):
            features_order = list(getattr(model, "feature_names_in_"))
            # pick intersection in same order
            features_order = [f for f in features_order if f in df.columns]
            df_for_pred = df[features_order] if features_order else df
        else:
            # fallback: use all columns in the incoming request
            features_order = list(df.columns)
            df_for_pred = df[features_order]

    # Prepare numpy array for model predict
    X_input = df_for_pred.values
    fv = X_input[0].tolist()
    fv_len = X_input.shape[1] if isinstance(X_input, (np.ndarray,)) else len(fv)

    # Helper to call model in threadpool (safely)
    loop = asyncio.get_running_loop()

    def sync_checks():
        out = {}
        try:
            # model outputs on input
            proba = None
            pred = None
            try:
                proba = model.predict_proba(X_input)
                pred = model.predict(X_input) if hasattr(model, "predict") else None
                out["input_proba"] = proba.tolist()
                out["input_pred"] = pred.tolist() if pred is not None else None
            except Exception as e:
                out["input_error"] = str(e)

            # Prepare test vectors sized to fv_len
            zeros = [[0.0] * fv_len]
            ones = [[1.0] * fv_len]
            # safe_scaled: try to fill with 0.9 except low-risk features (if known)
            safe = [0.9] * fv_len
            # attempt to lower velocity-like and fraud-count-like features if present
            for i, name in enumerate(features_order):
                lname = name.lower()
                if "velocity" in lname or "device_fraud" in lname or "fraud_count" in lname:
                    safe[i] = 0.0
            safe_scaled = [safe]

            raw_large = [[10000.0 if i == 0 else 750.0 if i == 1 else 45.0 if i == 2 else 0.0 for i in range(fv_len)]]

            tests = {
                "zeros": zeros,
                "ones": ones,
                "safe_scaled": safe_scaled,
                "raw_large": raw_large,
            }

            test_results = {}
            for name, tarr in tests.items():
                try:
                    p = model.predict_proba(np.array(tarr))
                    pr = model.predict(np.array(tarr)) if hasattr(model, "predict") else None
                    test_results[name] = {"proba": p.tolist(), "pred": pr.tolist() if pr is not None else None}
                except Exception as e:
                    test_results[name] = {"error": str(e)}

            out["tests"] = test_results

            # model feature names if present
            try:
                fn = getattr(model, "feature_names_", None) or getattr(model, "feature_names", None) or getattr(model, "feature_names_in_", None)
                out["model_feature_names"] = list(fn) if fn is not None else None
            except Exception:
                out["model_feature_names"] = None

            # feature importance if model has it
            try:
                if hasattr(model, "get_feature_importance"):
                    fi = model.get_feature_importance()
                    out["feature_importance"] = fi.tolist() if hasattr(fi, "tolist") else list(fi)
            except Exception as e:
                out["feature_importance_error"] = str(e)

            # some metadata if available
            try:
                out["model_type"] = type(model).__name__
            except Exception:
                out["model_type"] = None

        except Exception as e:
            out["error"] = str(e)
        return out

    out = await loop.run_in_executor(executor, sync_checks)

    return {
        "features_order_used": features_order,
        "input_fv": fv,
        "fv_len": fv_len,
        "model_outputs": out,
        "incoming_columns": list(df.columns),
    }


# include router and app startup instructions
app.include_router(router)

# Add this endpoint to main.py for debugging

@app.post("/debug/verify_features")
async def verify_features(transaction: Transaction):
    """
    Debug endpoint to verify feature alignment and check for preprocessing issues.
    Returns detailed information about feature values and model expectations.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    transaction_dict = transaction.model_dump()
    df = pd.DataFrame([transaction_dict])

    response = {
        "model_type": type(model).__name__,
        "expected_features_count": len(expected_features),
        "received_features_count": len(df.columns),
        "threshold": float(threshold),
        "expected_features": expected_features if expected_features else None,
        "received_features": list(df.columns),
        "missing_features": [f for f in expected_features if f not in df.columns] if expected_features else [],
        "extra_features": [f for f in df.columns if f not in expected_features] if expected_features else [],
    }

    # Feature value statistics
    if expected_features:
        aligned_df = df[[f for f in expected_features if f in df.columns]]
        response["feature_stats"] = {
            "min": float(aligned_df.values.min()),
            "max": float(aligned_df.values.max()),
            "mean": float(aligned_df.values.mean()),
            "std": float(aligned_df.values.std()),
        }
        
        # Check for suspicious values
        response["warnings"] = []
        if aligned_df.values.max() > 10:
            response["warnings"].append("Features contain large values (>10) - may not be properly scaled")
        if aligned_df.values.min() < -10:
            response["warnings"].append("Features contain very negative values (<-10) - may not be properly scaled")
        
        # Sample feature values
        response["sample_features"] = {
            name: float(value) 
            for name, value in aligned_df.iloc[0].head(10).items()
        }

    # Test prediction
    if expected_features and all(f in df.columns for f in expected_features):
        try:
            df_for_pred = df[expected_features]
            proba = model.predict_proba(df_for_pred)[0, 1]
            response["test_prediction"] = {
                "probability": float(proba),
                "is_fraud": bool(proba >= threshold),
                "risk_level": determine_risk_level(float(proba)),
            }
        except Exception as e:
            response["test_prediction_error"] = str(e)

    return response
# ---------------------------------------------------------------------
# Run with: python -m uvicorn fraud_detection_api.api.main:app --reload
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "fraud_detection_api.api.main:app", host="0.0.0.0", port=port, reload=True, log_level="info"
    )
