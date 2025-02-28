import asyncio
import logging
import os
import tempfile
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv
from fastmcp import Context

from knmi_weather_mcp.models import Coordinates, WeatherStation

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

    # Netherlands bounding box (mainland Netherlands)
    NL_BOUNDS = {
        "min_lat": 50.7,  # Southernmost point
        "max_lat": 53.7,  # Northernmost point
        "min_lon": 3.3,  # Westernmost point
        "max_lon": 7.2,  # Easternmost point
    }

    # Parameter mapping for 10-minute data
    PARAMETER_MAPPING = {
        "T": "temperature",  # Air temperature
        "RH": "relative_humidity",  # Relative humidity
        "FF": "wind_speed",  # 10-min mean wind speed
        "DD": "wind_direction",  # Wind direction
        "VIS": "visibility",  # Visibility
        "P": "air_pressure",  # Air pressure
        "RR": "precipitation_amount",  # Precipitation amount
        "DR": "precipitation_duration",  # Precipitation duration
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
        """Fetch current stations from KNMI Open Data API"""
        async with self._lock:
            try:
                async with httpx.AsyncClient() as client:
                    headers = {"Authorization": self._api_key}

                    # Get the latest file to extract station information
                    list_url = f"{self.BASE_URL}/datasets/{self.DATASET_NAME}/versions/{self.DATASET_VERSION}/files"
                    params = {"maxKeys": "1", "sorting": "desc", "order_by": "lastModified"}

                    logger.info("Fetching latest file for station information")
                    response = await client.get(list_url, headers=headers, params=params)
                    response.raise_for_status()

                    files_data = response.json()
                    if not files_data.get("files"):
                        raise ValueError("No data files available")

                    latest_file = files_data["files"][0]
                    filename = latest_file["filename"]

                    # Get download URL for the file
                    url_endpoint = f"{self.BASE_URL}/datasets/{self.DATASET_NAME}/versions/{self.DATASET_VERSION}/files/{filename}/url"  # noqa: E501
                    url_response = await client.get(url_endpoint, headers=headers)
                    url_response.raise_for_status()

                    download_url = url_response.json().get("temporaryDownloadUrl")
                    if not download_url:
                        raise ValueError("No download URL available")

                    # Download and read the NetCDF file to get station information
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_file = Path(temp_dir) / filename
                        file_response = await client.get(download_url)
                        file_response.raise_for_status()
                        temp_file.write_bytes(file_response.content)

                        with xr.open_dataset(temp_file) as ds:
                            new_stations = self._parse_stations(ds)

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

    def _parse_stations(self, ds: xr.Dataset) -> Dict[str, WeatherStation]:
        """Parse station information from NetCDF file"""
        new_stations = {}

        # Get station IDs and coordinates
        if "station" in ds.dims:
            station_ids = ds["station"].values
            lats = ds["lat"].values
            lons = ds["lon"].values

            # Log the raw data found
            logger.debug(f"Raw station IDs found: {station_ids}")
            logger.debug(f"Raw latitudes found: {lats}")
            logger.debug(f"Raw longitudes found: {lons}")
            logger.debug(f"Dataset structure: {ds}")
            logger.debug(f"Dataset dimensions: {ds.dims}")
            logger.debug(f"Dataset variables: {list(ds.variables)}")
            logger.debug(f"Dataset attributes: {ds.attrs}")

            # Convert station IDs to our format (remove '06' prefix)
            station_ids = [str(sid).replace("06", "") if str(sid).startswith("06") else str(sid) for sid in station_ids]
            logger.debug(f"Converted station IDs: {station_ids}")

            # Try to get station names from the dataset
            station_names = None
            name_variables = [
                "name",
                "station_name",
                "stn_name",
                "stationname",
                "station_names",
                "names",
                "stn_names",
                "stationnames",
                "NAMES",
                "NAME",
                "STN_NAME",
                "STATION_NAME",
            ]

            for name_var in name_variables:
                if name_var in ds.variables:
                    try:
                        station_names = ds[name_var].values
                        logger.info(f"Found station names in variable: {name_var}")
                        logger.debug(f"Raw station names: {station_names}")
                        break
                    except Exception as e:
                        logger.debug(f"Could not read station names from {name_var}: {e}")

            if station_names is None:
                # Try to find station names in dataset attributes
                for attr_name in ds.attrs:
                    if "name" in attr_name.lower() or "station" in attr_name.lower():
                        logger.debug(f"Found potential station name attribute: {attr_name}")
                        try:
                            attr_value = ds.attrs[attr_name]
                            if isinstance(attr_value, (str, bytes)):
                                logger.info(f"Found station names in attribute: {attr_name}")
                                station_names = [attr_value]
                            elif isinstance(attr_value, (list, np.ndarray)):
                                logger.info(f"Found station names array in attribute: {attr_name}")
                                station_names = attr_value
                            break
                        except Exception as e:
                            logger.debug(f"Could not read station names from attribute {attr_name}: {e}")

            logger.info(f"Found {len(station_ids)} stations in NetCDF file")
            logger.debug(f"Available dimensions: {ds.dims}")
            logger.debug(f"Available variables: {list(ds.variables)}")

            for i, station_id in enumerate(station_ids):
                station_id = str(station_id)
                coords = Coordinates(latitude=float(lats[i]), longitude=float(lons[i]))

                # Get station name from dataset
                station_name = None
                if station_names is not None:
                    try:
                        name = station_names[i] if i < len(station_names) else None
                        if isinstance(name, bytes):
                            name = name.decode("utf-8")
                        if isinstance(name, str) and name.strip():
                            station_name = name.strip()
                            logger.debug(f"Found name '{station_name}' for station {station_id}")
                    except Exception as e:
                        logger.debug(f"Could not decode station name for {station_id}: {e}")

                if not station_name:
                    station_name = f"Station {station_id}"
                    logger.debug(f"Using default name '{station_name}' for station {station_id}")

                # Log coordinates before validation
                logger.debug(f"Station {station_id} coordinates: lat={coords.latitude}, lon={coords.longitude}")

                # Only add station if coordinates are within mainland Netherlands
                if self._validate_coordinates(coords):
                    new_stations[station_id] = WeatherStation(
                        id=station_id,
                        name=station_name,
                        coordinates=coords,
                        elevation=0.0,  # Could be extracted if available
                        station_type="Weather",
                        region="Netherlands",
                    )
                    logger.debug(f"Added station {station_id}: {new_stations[station_id]}")
                else:
                    logger.debug(f"Station {station_id} coordinates outside Netherlands bounds")

        return new_stations

    def find_nearest_station(self, coords: Coordinates) -> WeatherStation:
        """Find the nearest weather station to given coordinates"""
        if not self._stations:
            raise ValueError("No stations available. Call refresh_stations first.")

        def calculate_distance(station: WeatherStation) -> float:
            """Calculate distance using Haversine formula"""
            # Convert coordinates to radians
            lat1, lon1 = radians(coords.latitude), radians(coords.longitude)
            lat2, lon2 = (
                radians(station.coordinates.latitude),
                radians(station.coordinates.longitude),
            )

            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))

            # Earth's radius in kilometers
            R = 6371.0

            return R * c

        # Find station with minimum distance
        nearest_station = min(self._stations.values(), key=calculate_distance)

        logger.info(f"Found nearest station: {nearest_station.name} ({nearest_station.id})")
        return nearest_station

    def _validate_coordinates(self, coords: Coordinates) -> bool:
        """Check if coordinates are within the Netherlands"""
        return (
            self.NL_BOUNDS["min_lat"] <= coords.latitude <= self.NL_BOUNDS["max_lat"]
            and self.NL_BOUNDS["min_lon"] <= coords.longitude <= self.NL_BOUNDS["max_lon"]
        )

    async def get_raw_station_data(self, station_id: str, ctx: Optional[Context] = None) -> Dict[str, Any]:
        """Get raw data for a specific station using KNMI Open Data API"""
        logger.info(f"Starting data fetch for station {station_id}")

        async with httpx.AsyncClient() as client:
            try:
                headers = {"Authorization": self._api_key}

                # Get station coordinates
                station = self._stations.get(station_id)
                if not station:
                    raise ValueError(f"Station {station_id} not found")

                # Validate coordinates are within bounds
                if not self._validate_coordinates(station.coordinates):
                    error_msg = (
                        f"Coordinates ({station.coordinates.latitude}, {station.coordinates.longitude}) are outside "
                        "valid bounds"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # List files endpoint with station filter
                list_url = f"{self.BASE_URL}/datasets/{self.DATASET_NAME}/versions/{self.DATASET_VERSION}/files"

                # Get the latest file for this station (sort by lastModified in descending order)
                params = {
                    "maxKeys": "1",
                    "sorting": "desc",
                    "order_by": "lastModified",
                    "station": station_id,  # Filter for specific station
                }

                logger.info(f"Requesting latest 10-minute data for station {station_id}")
                logger.debug(f"Query parameters: {params}")

                # Get the latest file metadata
                response = await client.get(list_url, headers=headers, params=params)
                response.raise_for_status()

                files_data = response.json()
                if not files_data.get("files"):
                    raise ValueError(f"No data files available for station {station_id}")

                latest_file = files_data["files"][0]
                filename = latest_file["filename"]

                # Get download URL for the file
                url_endpoint = (
                    f"{self.BASE_URL}/datasets/{self.DATASET_NAME}/versions/{self.DATASET_VERSION}/files/{filename}/url"
                )
                url_response = await client.get(url_endpoint, headers=headers)
                url_response.raise_for_status()

                download_url = url_response.json().get("temporaryDownloadUrl")
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

                            # Verify we have the correct station data
                            if "station" in ds.dims:
                                stations_in_file = ds["station"].values
                                logger.debug(f"Stations in file: {stations_in_file}")

                                # Convert our station_id to match KNMI format (add '06' prefix if needed)
                                knmi_station_id = f"06{station_id}" if not station_id.startswith("06") else station_id
                                logger.debug(f"Looking for station ID {knmi_station_id} (original: {station_id})")

                                # Convert to the same type as in the file
                                file_station_type = type(stations_in_file[0])
                                comparable_station_id = file_station_type(knmi_station_id)

                                if comparable_station_id not in stations_in_file:
                                    raise ValueError(
                                        f"Station {station_id} (as {knmi_station_id}) not found in file. "
                                        f"Available stations: {stations_in_file}"
                                    )

                                # Find the index of our station
                                station_idx = np.where(stations_in_file == comparable_station_id)[0]
                                if len(station_idx) == 0:
                                    raise ValueError(f"Could not find index for station {knmi_station_id}")
                                station_idx = station_idx[0]
                                logger.debug(f"Found station {knmi_station_id} at index {station_idx}")
                            else:
                                raise ValueError("No station dimension found in file")

                            # Create a dictionary to store the measurements
                            measurements = {}

                            # Map NetCDF variables to our model fields
                            param_mapping = {
                                "ta": "temperature",  # Air temperature (°C)
                                "rh": "relative_humidity",  # Relative humidity (%)
                                "ff": "wind_speed",  # Wind speed (m/s)
                                "dd": "wind_direction",  # Wind direction (degrees)
                                "vis": "visibility",  # Visibility (meters)
                                "pp": "air_pressure",  # Air pressure (hPa)
                                "rr": "precipitation_amount",  # Precipitation amount (mm)
                                "dr": "precipitation_duration",  # Precipitation duration (minutes)
                            }

                            # Extract values for each parameter
                            for nc_param, model_field in param_mapping.items():
                                if nc_param in ds.variables:
                                    var = ds[nc_param]
                                    logger.debug(f"Found variable {nc_param} with dimensions {var.dims}")

                                    try:
                                        # Get the value for our specific station
                                        if "station" in var.dims:
                                            # Select our station first
                                            value = var.isel(station=station_idx)

                                            # Then get the latest time if it exists
                                            if "time" in value.dims:
                                                value = value.isel(time=-1)

                                            # Handle any remaining dimensions
                                            while len(value.dims) > 0:
                                                value = value.isel({value.dims[0]: 0})

                                            value = float(value.values)
                                        else:
                                            logger.warning(f"Variable {nc_param} does not have a station dimension")
                                            continue

                                        if not np.isnan(value):
                                            measurements[model_field] = value
                                            logger.debug(f"Got {model_field} = {value} for station {station_id}")
                                    except Exception as e:
                                        logger.warning(f"Could not extract value for {nc_param}: {e}")

                            # Get the timestamp from the time variable if it exists
                            timestamp = latest_file.get("lastModified")
                            if "time" in ds.variables:
                                try:
                                    time_var = ds["time"]
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
                                "measurements": measurements,
                                "metadata": {
                                    "station_id": station_id,
                                    "station_name": station.name,
                                    "timestamp": timestamp,
                                    "filename": filename,
                                    "variables": list(
                                        ds.variables.keys()
                                    ),  # Add list of available variables for debugging
                                },
                            }

                    except Exception as e:
                        logger.error(f"Error reading NetCDF file: {e}")
                        # Try to read the file content to debug
                        logger.debug(f"File content (first 1000 bytes): {file_response.content[:1000]}")
                        raise ValueError(f"Failed to read NetCDF file: {str(e)}") from e

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
        region="Utrecht",
    ),
    "240": WeatherStation(
        id="240",
        name="Amsterdam Schiphol",
        coordinates=Coordinates(latitude=52.318, longitude=4.790),
        elevation=-3.0,
        station_type="Airport",
        region="Noord-Holland",
    ),
}
