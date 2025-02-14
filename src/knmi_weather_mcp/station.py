from typing import Dict, Optional, Any
import asyncio
import httpx
from datetime import datetime
from fastmcp import Context
from knmi_weather_mcp.models import WeatherStation, Coordinates
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import tempfile
import xarray as xr
import numpy as np
import pandas as pd


load_dotenv()

# Get logger for this module
logger = logging.getLogger("knmi_weather.station")

class StationManager:
    """Manages KNMI weather stations"""
    _instance = None
    _initialized = False
    
    # API endpoints
    BASE_URL = "https://api.dataplatform.knmi.nl/open-data/v1"
    DATASET_NAME = "Actuele10mindataKNMIstations"
    DATASET_VERSION = "2"
    
    # Netherlands bounding box (from dataset metadata)
    NL_BOUNDS = {
        'min_lat': 12.0,      # Updated based on dataset bounds
        'max_lat': 55.7,      # Including BES islands
        'min_lon': -68.5,     # And North Sea platforms
        'max_lon': 7.4
    }
    
    # Parameter mapping for 10-minute data
    PARAMETER_MAPPING = {
        'T': 'temperature',           # Air temperature
        'RH': 'relative_humidity',    # Relative humidity
        'FF': 'wind_speed',          # 10-min mean wind speed
        'DD': 'wind_direction',       # Wind direction
        'VIS': 'visibility',         # Visibility
        'P': 'air_pressure',         # Air pressure
        'RR': 'precipitation_amount', # Precipitation amount
        'DR': 'precipitation_duration' # Precipitation duration
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not StationManager._initialized:
            self._stations: Dict[str, WeatherStation] = {}
            self._last_update: Optional[datetime] = None
            self._lock = asyncio.Lock()
            self._api_key = os.getenv("KNMI_API_KEY")
            
            if not self._api_key:
                logger.error("KNMI_API_KEY environment variable is missing")
                raise ValueError("KNMI_API_KEY environment variable is required")
                
            # Strip "Bearer" prefix if it exists in the env var
            self._api_key = self._api_key.replace("Bearer ", "")
            logger.info("StationManager initialized successfully")
            StationManager._initialized = True

    @property
    def stations(self) -> Dict[str, WeatherStation]:
        return self._stations

    async def refresh_stations(self, ctx: Optional[Context] = None) -> None:
        """Fetch current stations from KNMI EDR API"""
        async with self._lock:
            try:
                async with httpx.AsyncClient() as client:
                    headers = {'Authorization': self._api_key}
                    
                    # Test EDR API capabilities first
                    capabilities_url = f"{self.BASE_URL}/collections/{self.DATASET_NAME}"
                    logger.info("Testing EDR API capabilities")
                    
                    capabilities_response = await client.get(capabilities_url, headers=headers)
                    capabilities_response.raise_for_status()
                    
                    capabilities = capabilities_response.json()
                    logger.debug(f"EDR API capabilities: {capabilities}")
                    
                    # Get locations from EDR API
                    locations_url = f"{capabilities_url}/locations"
                    logger.info(f"Fetching stations from EDR API")
                    
                    response = await client.get(locations_url, headers=headers)
                    
                    if response.status_code == 401:
                        logger.error("Authentication failed. Please check your API key.")
                        raise ValueError("Authentication failed with KNMI API")
                        
                    response.raise_for_status()
                    
                    locations_data = response.json()
                    logger.info(f"Received {len(locations_data.get('features', []))} stations")
                    logger.debug(f"Raw locations data: {locations_data}")
                    
                    new_stations = {}
                    for feature in locations_data.get('features', []):
                        properties = feature.get('properties', {})
                        geometry = feature.get('geometry', {})
                        coordinates = geometry.get('coordinates', [])
                        
                        if len(coordinates) >= 2:
                            # Get station ID, trying multiple possible property names
                            station_id = str(properties.get('id') or 
                                           properties.get('stationId') or 
                                           properties.get('station_id') or 
                                           properties.get('code', ''))
                            
                            # Only add station if we have a valid ID and coordinates are within Netherlands
                            if station_id and coordinates[0] and coordinates[1]:
                                coords = Coordinates(
                                    # GeoJSON uses [longitude, latitude] order
                                    longitude=coordinates[0],
                                    latitude=coordinates[1]
                                )
                                
                                # Validate coordinates are within Netherlands
                                if self._validate_coordinates(coords):
                                    new_stations[station_id] = WeatherStation(
                                        id=station_id,
                                        name=properties.get('name', f'Station {station_id}'),
                                        coordinates=coords,
                                        elevation=properties.get('elevation', 0.0),
                                        station_type=properties.get('type'),
                                        region=properties.get('region')
                                    )
                                    logger.debug(f"Added station {station_id}: {new_stations[station_id]}")
                    
                    if not new_stations:
                        logger.warning("No valid stations found in Netherlands, using fallback stations")
                        new_stations = FALLBACK_STATIONS
                    
                    self._stations = new_stations
                    self._last_update = datetime.now()
                    logger.info(f"Successfully loaded {len(self._stations)} stations")

            except Exception as e:
                logger.error(f"Failed to refresh stations: {str(e)}")
                if not self._stations:
                    self._stations = FALLBACK_STATIONS

    def find_nearest_station(self, coords: Coordinates) -> WeatherStation:
        """Find the nearest weather station to given coordinates"""
        if not self._stations:
            raise ValueError("No stations available. Call refresh_stations first.")
            
        def calculate_distance(station: WeatherStation) -> float:
            """Calculate distance using Haversine formula"""
            # Convert coordinates to radians
            lat1, lon1 = radians(coords.latitude), radians(coords.longitude)
            lat2, lon2 = radians(station.coordinates.latitude), radians(station.coordinates.longitude)
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            
            # Earth's radius in kilometers
            R = 6371.0
            
            return R * c
        
        # Find station with minimum distance
        nearest_station = min(
            self._stations.values(),
            key=calculate_distance
        )
        
        logger.info(f"Found nearest station: {nearest_station.name} ({nearest_station.id})")
        return nearest_station

    def _validate_coordinates(self, coords: Coordinates) -> bool:
        """Check if coordinates are within the Netherlands"""
        return (
            self.NL_BOUNDS['min_lat'] <= coords.latitude <= self.NL_BOUNDS['max_lat'] and
            self.NL_BOUNDS['min_lon'] <= coords.longitude <= self.NL_BOUNDS['max_lon']
        )

    async def get_raw_station_data(self, station_id: str, ctx: Optional[Context] = None) -> Dict[str, Any]:
        """Get raw data for a specific station using KNMI Open Data API"""
        logger.info(f"Starting data fetch for station {station_id}")

        async with httpx.AsyncClient() as client:
            try:
                headers = {'Authorization': self._api_key}
                
                # Get station coordinates
                station = self._stations.get(station_id)
                if not station:
                    raise ValueError(f"Station {station_id} not found")
                
                # Validate coordinates are within bounds
                if not self._validate_coordinates(station.coordinates):
                    error_msg = f"Coordinates ({station.coordinates.latitude}, {station.coordinates.longitude}) are outside valid bounds"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # List files endpoint with station filter
                list_url = f"{self.BASE_URL}/datasets/{self.DATASET_NAME}/versions/{self.DATASET_VERSION}/files"
                
                # Get the latest file for this station (sort by lastModified in descending order)
                params = {
                    'maxKeys': '1',
                    'sorting': 'desc',
                    'order_by': 'lastModified',
                    'station': station_id  # Filter for specific station
                }
                
                logger.info(f"Requesting latest 10-minute data for station {station_id}")
                logger.debug(f"Query parameters: {params}")
                
                # Get the latest file metadata
                response = await client.get(list_url, headers=headers, params=params)
                response.raise_for_status()
                
                files_data = response.json()
                if not files_data.get('files'):
                    raise ValueError(f"No data files available for station {station_id}")
                
                latest_file = files_data['files'][0]
                filename = latest_file['filename']
                
                # Get download URL for the file
                url_endpoint = f"{self.BASE_URL}/datasets/{self.DATASET_NAME}/versions/{self.DATASET_VERSION}/files/{filename}/url"
                url_response = await client.get(url_endpoint, headers=headers)
                url_response.raise_for_status()
                
                download_url = url_response.json().get('temporaryDownloadUrl')
                if not download_url:
                    raise ValueError("No download URL available")
                
                # Create a temporary directory to store the NetCDF file
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file = Path(temp_dir) / filename
                    
                    # Download the file
                    logger.info(f"Downloading file: {filename}")
                    file_response = await client.get(download_url)
                    file_response.raise_for_status()
                    
                    # Save the binary content
                    temp_file.write_bytes(file_response.content)
                    
                    # Open and read the NetCDF file
                    logger.info("Reading NetCDF file")
                    try:
                        with xr.open_dataset(temp_file) as ds:
                            # Log the structure of the dataset
                            logger.debug(f"NetCDF structure: {ds}")
                            logger.debug(f"Available variables: {list(ds.variables)}")
                            logger.debug(f"Dimensions: {ds.dims}")
                            
                            # Create a dictionary to store the measurements
                            measurements = {}
                            
                            # Map NetCDF variables to our model fields
                            param_mapping = {
                                'ta': 'temperature',           # Air temperature (°C)
                                'rh': 'relative_humidity',     # Relative humidity (%)
                                'ff': 'wind_speed',           # Wind speed (m/s)
                                'dd': 'wind_direction',       # Wind direction (degrees)
                                'vis': 'visibility',          # Visibility (meters)
                                'pp': 'air_pressure',         # Air pressure (hPa)
                                'rr': 'precipitation_amount',  # Precipitation amount (mm)
                                'dr': 'precipitation_duration' # Precipitation duration (minutes)
                            }
                            
                            # Extract values for each parameter
                            for nc_param, model_field in param_mapping.items():
                                if nc_param in ds.variables:
                                    var = ds[nc_param]
                                    logger.debug(f"Found variable {nc_param} with dimensions {var.dims}")
                                    
                                    try:
                                        # Get the latest value
                                        if 'time' in var.dims:
                                            # Get the last time index
                                            value = var.isel(time=-1)
                                            # If there are other dimensions, take the first value
                                            while len(value.dims) > 0:
                                                value = value.isel({value.dims[0]: 0})
                                            value = float(value.values)
                                        else:
                                            # Single value
                                            value = var.values
                                            # If multi-dimensional, take the first value
                                            while isinstance(value, np.ndarray) and value.size > 1:
                                                value = value[0]
                                            value = float(value)
                                            
                                        if not np.isnan(value):
                                            measurements[model_field] = value
                                    except Exception as e:
                                        logger.warning(f"Could not extract value for {nc_param}: {e}")
                            
                            # Get the timestamp from the time variable if it exists
                            timestamp = latest_file.get('lastModified')
                            if 'time' in ds.variables:
                                try:
                                    time_var = ds['time']
                                    # Get the latest timestamp
                                    time_value = time_var.isel(time=-1)
                                    # If multi-dimensional, take the first value
                                    while len(time_value.dims) > 0:
                                        time_value = time_value.isel({time_value.dims[0]: 0})
                                    timestamp = pd.Timestamp(time_value.values).isoformat()
                                except Exception as e:
                                    logger.warning(f"Could not parse time variable: {e}")
                                    logger.debug(f"Time variable structure: {time_var}")
                            
                            return {
                                'measurements': measurements,
                                'metadata': {
                                    'station_id': station_id,
                                    'station_name': station.name,
                                    'timestamp': timestamp,
                                    'filename': filename,
                                    'variables': list(ds.variables.keys())  # Add list of available variables for debugging
                                }
                            }
                            
                    except Exception as e:
                        logger.error(f"Error reading NetCDF file: {e}")
                        # Try to read the file content to debug
                        logger.debug(f"File content (first 1000 bytes): {file_response.content[:1000]}")
                        raise ValueError(f"Failed to read NetCDF file: {str(e)}")

            except Exception as e:
                logger.error(f"Error in get_raw_station_data: {str(e)}")
                if isinstance(e, httpx.HTTPError):
                    logger.error(f"HTTP Error details: {str(e)}")
                    logger.error(f"Request URL: {e.request.url if e.request else 'Unknown'}")
                raise

# Default fallback stations
FALLBACK_STATIONS = {
    "260": WeatherStation(
        id="260",
        name="De Bilt",
        coordinates=Coordinates(latitude=52.101, longitude=5.177),
        elevation=2.0,
        station_type="Main",
        region="Utrecht"
    ),
    "240": WeatherStation(
        id="240",
        name="Amsterdam Schiphol",
        coordinates=Coordinates(latitude=52.318, longitude=4.790),
        elevation=-3.0,
        station_type="Airport",
        region="Noord-Holland"
    )
}