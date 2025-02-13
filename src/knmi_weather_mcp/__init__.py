"""KNMI Weather MCP Server package."""

__version__ = "0.1.0"

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class Coordinates(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

class WeatherStation(BaseModel):
    id: str
    name: str
    coordinates: Coordinates
    elevation: float
    station_type: Optional[str] = None
    region: Optional[str] = None

class WeatherData(BaseModel):
    temperature: float
    humidity: float
    condition: str
    timestamp: datetime
    station_name: str
    station_id: str
