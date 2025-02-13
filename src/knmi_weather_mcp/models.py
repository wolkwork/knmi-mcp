from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime

class Coordinates(BaseModel):
    """Geographic coordinates"""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

class WeatherStation(BaseModel):
    """KNMI weather station"""
    id: str
    name: str
    coordinates: Coordinates
    elevation: float
    station_type: Optional[str] = None
    region: Optional[str] = None

class WeatherData(BaseModel):
    """Weather measurement data"""
    temperature: float
    humidity: float
    timestamp: datetime
    station_name: str
    station_id: str
    wind_speed: Optional[float] = None
    wind_direction: Optional[float] = None
    precipitation: Optional[float] = None
    visibility: Optional[float] = None
    pressure: Optional[float] = None

class StationData(BaseModel):
    """Raw station measurement data"""
    measurements: Dict[str, float]
    metadata: Dict[str, str]
    timestamp: datetime
