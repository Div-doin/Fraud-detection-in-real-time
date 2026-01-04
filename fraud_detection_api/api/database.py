# fraud_detection_api/api/database.py
"""
Async MongoDB helper for FastAPI using motor.
Provides:
 - connect_to_mongo() / close_mongo_connection()
 - get_db() FastAPI dependency (yields AsyncIOMotorDatabase or None)
 - get_mongo_uri_info(), is_connected()
 - exported collection names/constants
"""

from __future__ import annotations
import os
import asyncio
import logging
from typing import AsyncGenerator, Optional, Dict
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ServerSelectionTimeoutError, PyMongoError

logger = logging.getLogger(__name__)

# Environment variables (names chosen to match main.py expectations)
MONGO_URI: str = os.getenv("MONGO_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
DB_NAME: str = os.getenv("DB_NAME", os.getenv("MONGO_DB_NAME", "fraud_detection_db"))

MONGO_PREDICTION_COLLECTION: str = os.getenv("MONGO_PREDICTION_COLLECTION", "predictions")
MONGO_ALERTS_COLLECTION: str = os.getenv("MONGO_ALERTS_COLLECTION", "alerts")

# Module-level client / db references
_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None

def get_mongo_uri_info() -> Dict[str, str]:
    """Return connection info (used by main.py debug/info endpoints)."""
    return {
        "uri": MONGO_URI,
        "db_name": DB_NAME,
        "prediction_collection": MONGO_PREDICTION_COLLECTION,
        "alerts_collection": MONGO_ALERTS_COLLECTION,
    }

def is_connected() -> bool:
    """Quick (sync) check if client object exists and db is assigned."""
    return _client is not None and _db is not None

async def connect_to_mongo(max_retries: int = 5, delay_seconds: float = 1.0) -> None:
    """
    Create AsyncIOMotorClient and verify connection with exponential backoff.
    Raises RuntimeError if unable to connect after retries.
    """
    global _client, _db

    if _client is not None and _db is not None:
        logger.debug("Mongo already connected")
        return

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            logger.info("Attempting MongoDB connection to %s (attempt %d/%d)", MONGO_URI, attempt, max_retries)
            _client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            # reliable ping: use admin command
            await _client.admin.command("ping")
            _db = _client[DB_NAME]
            logger.info("✅ MongoDB connected to %s, DB=%s", MONGO_URI, DB_NAME)
            return
        except (ServerSelectionTimeoutError, PyMongoError) as e:
            logger.warning("Mongo connect attempt %d failed: %s", attempt, e)
            # close partially-open client to ensure fresh retry
            try:
                if _client is not None:
                    _client.close()
            except Exception:
                pass
            _client = None
            _db = None
            # backoff
            await asyncio.sleep(delay_seconds * (2 ** (attempt - 1)))
        except Exception as e:
            logger.exception("Unexpected error while connecting to Mongo: %s", e)
            _client = None
            _db = None
            await asyncio.sleep(delay_seconds * (2 ** (attempt - 1)))

    raise RuntimeError(f"Could not connect to MongoDB at {MONGO_URI} after {max_retries} attempts")

async def close_mongo_connection() -> None:
    """Close motor client cleanly."""
    global _client, _db
    if _client is not None:
        try:
            _client.close()
            logger.info("MongoDB connection closed")
        except Exception as e:
            logger.warning("Error closing MongoDB client: %s", e)
    _client = None
    _db = None

async def get_db() -> AsyncGenerator[Optional[AsyncIOMotorDatabase], None]:
    """
    FastAPI dependency that yields an AsyncIOMotorDatabase instance or None.
    Usage: async def endpoint(db = Depends(get_db)): ...
    """
    global _db
    if _db is None:
        try:
            await connect_to_mongo()
        except Exception as e:
            logger.warning("Could not connect to Mongo in get_db(): %s", e)
            yield None
            return

    yield _db

# export names expected by main.py
__all__ = [
    "get_db",
    "connect_to_mongo",
    "close_mongo_connection",
    "get_mongo_uri_info",
    "MONGO_PREDICTION_COLLECTION",
    "MONGO_ALERTS_COLLECTION",
    "is_connected",
]
