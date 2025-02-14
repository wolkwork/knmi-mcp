from fastmcp import FastMCP, Context
from typing import Dict, List, Any
import httpx
from knmi_weather_mcp.models import WeatherStation, WeatherData, Coordinates
from knmi_weather_mcp.station import StationManager
from knmi_weather_mcp.location import get_coordinates
from knmi_weather_mcp.weather import WeatherService
from dotenv import load_dotenv
import os
import logging
from pathlib import Path

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will still print to console
    ]
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
    debug=True,
    log_level="DEBUG",
    logger=logger,
    context_class=LoggingContext,
    python_path=[str(src_dir)]  # Use the dynamically determined src directory
)

# Initialize station manager
station_manager = StationManager()

# Initialize weather service
weather_service = WeatherService()

# Tools
@mcp.tool()
async def test_logging(ctx: Context) -> Dict[str, Any]:
    """Test if logging is working"""
    logger.error("Test ERROR message")  # Should always show
    logger.info("Test INFO message")
    logger.debug("Test DEBUG message")
    return {"status": "completed", "message": "Logging test completed"}

@mcp.tool()
async def test_api_key(ctx: Context) -> Dict[str, Any]:
    """Test KNMI API key"""
    api_key = os.getenv("KNMI_API_KEY")
    async with httpx.AsyncClient() as client:
        # Test EDR API endpoint
        response = await client.get(
            "https://api.dataplatform.knmi.nl/edr/v1/collections/observations/locations",
            headers={'Authorization': api_key}
        )
        
        result = {
            "status_code": response.status_code,
            "is_authorized": response.status_code == 200,
            "endpoint": "EDR API - Locations endpoint",
        }
        
        if response.status_code == 200:
            data = response.json()
            result["stations_count"] = len(data.get('features', []))
            result["response_data"] = data
        else:
            result["error"] = response.text
            
        return result

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
            raise ValueError(f"Location '{location}' ({coords.latitude}, {coords.longitude}) is outside the Netherlands. This tool only works for locations within the Netherlands.")
        
        logger.info("Step 3: Finding nearest station")
        station = station_manager.find_nearest_station(coords)
        logger.info(f"Using station: {station.name} ({station.id})")
        
        logger.info("Step 4: Getting weather data")
        weather_data = await station_manager.get_raw_station_data(station.id, ctx)
        
        logger.info("Weather data retrieved successfully")
        return weather_data
        
    except Exception as e:
        logger.error(f"Error getting weather: {str(e)}")
        raise

@mcp.tool()
async def search_location(
    query: str,
    ctx: Context
) -> List[Dict[str, str]]:
    """
    Search for locations in the Netherlands
    
    Args:
        query: Search term for location
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                'q': f"{query}, Netherlands",
                'format': 'json',
                'limit': 5
            },
            headers={'User-Agent': 'KNMI_Weather_MCP/1.0'}
        )
        response.raise_for_status()
        
        results = []
        for place in response.json():
            results.append({
                'name': place['display_name'],
                'type': place['type'],
                'latitude': place['lat'],
                'longitude': place['lon']
            })
            
        return results

@mcp.tool()
async def get_nearest_station(
    latitude: float,
    longitude: float,
    ctx: Context
) -> WeatherStation:
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
async def what_is_the_weather_like_in(
    location: str,
    ctx: Context
) -> str:
    """
    Get and interpret weather data for a location in the Netherlands
    
    Args:
        location: City or place name in the Netherlands
    Returns:
        A natural language interpretation of the current weather conditions
    """
    # Get the weather data
    weather_data = await weather_service.get_weather_by_location(location, ctx)
    
    # Convert to dict and ensure all fields are present
    data_dict = weather_data.dict()
    
    # Use the interpretation prompt to analyze it
    return weather_interpretation(data_dict)

# Prompts
@mcp.prompt()
def weather_interpretation(raw_data: Dict[str, Any]) -> str:
    """Help Claude interpret raw weather data"""
    return f"""Please analyze this weather data from KNMI and provide:
    1. A clear summary of current conditions
    2. Important weather measurements and their values
    3. Any notable patterns or extreme values
    4. Relevant clothing advice based on the conditions
    
    Current weather data from {raw_data['station_name']} ({raw_data['station_id']}) at {raw_data['timestamp']}:
    - Temperature: {raw_data['temperature']}°C
    - Humidity: {raw_data['humidity']}%
    - Wind Speed: {raw_data['wind_speed']} m/s
    - Wind Direction: {raw_data['wind_direction']} degrees
    - Precipitation: {raw_data['precipitation']} mm
    - Visibility: {raw_data['visibility']} meters
    - Pressure: {raw_data['pressure']} hPa
    """

if __name__ == "__main__":
    # For development testing
    import uvicorn
    uvicorn.run(mcp.app, host="localhost", port=8000)