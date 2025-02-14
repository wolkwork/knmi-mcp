from typing import Dict, Optional
from datetime import datetime
from fastmcp import Context
from knmi_weather_mcp.models import WeatherData, Coordinates
from knmi_weather_mcp.station import StationManager
from knmi_weather_mcp.location import get_coordinates

class WeatherService:
    """Service for fetching and processing KNMI weather data"""
    
    def __init__(self):
        self.station_manager = StationManager()
        
    async def get_weather_by_location(self, location: str, ctx: Optional[Context] = None) -> WeatherData:
        """Get weather data for a location"""
        # Ensure we have a context
        if not ctx:
            raise ValueError("Context is required for logging")
            
        ctx.info(f"=== Weather Lookup Process for {location} ===")
        
        try:
            # Get coordinates for location
            ctx.info("Step 1: Getting coordinates")
            coords = await get_coordinates(location)
            ctx.info(f"Found coordinates: {coords}")
                
            # Find nearest station
            ctx.info("Step 2: Finding nearest station")
            await self.station_manager.refresh_stations(ctx)
            station = self.station_manager.find_nearest_station(coords)
            ctx.info(f"Using station {station.name} ({station.id}) for {location}")
                
            # Get weather data
            ctx.info("Step 3: Getting weather data")
            station_data = await self.station_manager.get_raw_station_data(station.id, ctx)
            ctx.info(f"Raw station data: {station_data}")
            
            # Create WeatherData object
            ctx.info("Step 4: Processing weather data")
            measurements = station_data.get('measurements', {})
            metadata = station_data.get('metadata', {})
            
            # Create a dictionary with default values
            weather_data_dict = {
                'temperature': measurements.get('temperature', 0.0),
                'humidity': measurements.get('relative_humidity', 0.0),  # Map from relative_humidity
                'wind_speed': measurements.get('wind_speed', 0.0),
                'wind_direction': measurements.get('wind_direction', 0.0),
                'precipitation': measurements.get('precipitation_amount', 0.0),
                'visibility': measurements.get('visibility', 0.0),
                'pressure': measurements.get('air_pressure', 0.0),
                'timestamp': metadata.get('timestamp', datetime.now()),
                'station_id': metadata.get('station_id', station.id),
                'station_name': metadata.get('station_name', station.name)
            }
            
            weather_data = WeatherData(**weather_data_dict)
            ctx.info(f"Successfully created weather data object: {weather_data}")
            return weather_data
                
        except Exception as e:
            ctx.error(f"Failed to get weather data: {str(e)}")
            raise 