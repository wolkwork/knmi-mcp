import logging
from pathlib import Path
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp.server import Context, FastMCP

from knmi_weather_mcp.location import get_coordinates
from knmi_weather_mcp.models import Coordinates, WeatherStation
from knmi_weather_mcp.station import StationManager
from knmi_weather_mcp.weather import WeatherService

# Get the absolute path to the src directory
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent.parent

load_dotenv()

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "knmi_weather.log"

# Configure logging with a more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),  # This will still print to console
    ],
)

logger = logging.getLogger("knmi_weather")


# Create a custom context class that writes to our logger
class LoggingContext(Context):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("knmi_weather.context")

    async def debug(self, message: str) -> None:
        self.logger.debug(message)
        await super().debug(message)

    async def info(self, message: str) -> None:
        self.logger.info(message)
        await super().info(message)

    async def error(self, message: str) -> None:
        self.logger.error(message)
        await super().error(message)


# Initialize FastMCP server with our custom context class
mcp = FastMCP(
    "KNMI Weather",
    description="Raw KNMI weather data provider for the Netherlands",
    dependencies=["httpx", "pydantic", "python-dotenv", "pandas", "xarray", "numpy", "netCDF4"],
    debug=False,
    log_level="INFO",
    logger=logger,
    context_class=LoggingContext,
    python_path=[str(src_dir)],  # Use the dynamically determined src directory
    port=8001,
)

# Initialize station manager
station_manager = StationManager()

# Initialize weather service
weather_service = WeatherService()


# Tools
@mcp.tool()
async def get_location_weather(location: str, ctx: Context) -> Dict[str, Any]:
    """Get current weather data for a location"""
    logger.info(f"Starting weather request for {location}")

    try:
        # Log each step
        logger.info("Step 1: Refreshing stations")
        await station_manager.refresh_stations(ctx)

        logger.info("Step 2: Getting coordinates")
        coords = await get_coordinates(location)
        logger.debug(f"Coordinates found: {coords}")

        # Check if coordinates are within Netherlands
        if not station_manager._validate_coordinates(coords):
            raise ValueError(
                f"Location '{location}' ({coords.latitude}, {coords.longitude}) is outside the Netherlands. This tool only works for locations within the Netherlands."
            )

        logger.info("Step 3: Finding nearest station")
        station = station_manager.find_nearest_station(coords)
        logger.info(f"Using station: {station.name} ({station.id})")

        logger.info("Step 4: Getting weather data")
        weather_data = await station_manager.get_raw_station_data(station.id, ctx)

        logger.info("Weather data retrieved successfully")
        return weather_data

    except Exception as e:
        logger.error(f"Error getting weather: {str(e)}")
        return f"Error: Unable to get weather data for {location}. {str(e)}"


@mcp.tool()
async def search_location(query: str, ctx: Context) -> List[Dict[str, str]]:
    """
    Search for locations in the Netherlands

    Args:
        query: Search term for location
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{query}, Netherlands", "format": "json", "limit": 5},
            headers={"User-Agent": "KNMI_Weather_MCP/1.0"},
        )
        response.raise_for_status()

        results = []
        for place in response.json():
            results.append(
                {
                    "name": place["display_name"],
                    "type": place["type"],
                    "latitude": place["lat"],
                    "longitude": place["lon"],
                }
            )

        return results


@mcp.tool()
async def get_nearest_station(latitude: float, longitude: float, ctx: Context) -> WeatherStation:
    """
    Find the nearest KNMI weather station to given coordinates

    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
    """
    await station_manager.refresh_stations(ctx)
    coords = Coordinates(latitude=latitude, longitude=longitude)
    return station_manager.find_nearest_station(coords)


@mcp.tool()
async def what_is_the_weather_like_in(location: str, ctx: Context) -> str:
    """
    Get and interpret weather data for a location in the Netherlands

    Args:
        location: City or place name in the Netherlands
    Returns:
        A natural language interpretation of the current weather conditions
    """
    try:
        # Get the coordinates for the location
        coords = await get_coordinates(location)

        # Get the weather data
        weather_data = await weather_service.get_weather_by_location(location, ctx)

        # Convert to dict and ensure all fields are present
        data_dict = weather_data.dict()

        # Add location information
        data_dict["requested_location"] = location
        data_dict["location_coordinates"] = {
            "latitude": coords.latitude,
            "longitude": coords.longitude,
        }

        # Use the interpretation prompt to analyze it
        return weather_interpretation(data_dict)
    except Exception as e:
        logger.error(f"Error getting weather for {location}: {str(e)}")
        return f"Error: Unable to get weather data for {location}. {str(e)}"


# Prompts
@mcp.prompt()
def weather_interpretation(raw_data: Dict[str, Any]) -> str:
    """Help Claude interpret raw weather data"""
    try:
        location = raw_data.get("requested_location", "Unknown location")
        coords = raw_data.get("location_coordinates", {})
        lat = coords.get("latitude", 0.0)
        lon = coords.get("longitude", 0.0)
        station_name = raw_data.get("station_name", "Unknown station")
        station_id = raw_data.get("station_id", "Unknown ID")
        timestamp = raw_data.get("timestamp", "Unknown time")

        return f"""Please analyze this weather data from KNMI and provide:
        1. A clear summary of current conditions
        2. Important weather measurements and their values
        3. Any notable patterns or extreme values
        4. Relevant clothing advice based on the conditions
        
        Location: {location} ({lat:.3f}°N, {lon:.3f}°E)
        Weather station: {station_name} ({station_id}) at {timestamp}
        
        Current measurements:
        - Temperature: {raw_data.get("temperature", "N/A")}°C
        - Humidity: {raw_data.get("humidity", "N/A")}%
        - Wind Speed: {raw_data.get("wind_speed", "N/A")} m/s
        - Wind Direction: {raw_data.get("wind_direction", "N/A")} degrees
        - Precipitation: {raw_data.get("precipitation", "N/A")} mm
        - Visibility: {raw_data.get("visibility", "N/A")} meters
        - Pressure: {raw_data.get("pressure", "N/A")} hPa
        """
    except Exception as e:
        logger.error(f"Error formatting weather interpretation: {str(e)}")
        return "Error: Unable to interpret weather data due to missing or invalid data."


if __name__ == "__main__":
    # For running directly
    mcp.run()
