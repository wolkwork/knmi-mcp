import logging

import httpx

from knmi_weather_mcp.models import Coordinates

logger = logging.getLogger("knmi_weather.location")


async def get_coordinates(location: str) -> Coordinates:
    """Get coordinates for a location using OpenStreetMap Nominatim"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": f"{location}, Netherlands",  # Force Netherlands search
                    "format": "json",
                    "limit": 1,
                    "countrycodes": "nl",  # Restrict to Netherlands
                },
                headers={"User-Agent": "KNMI_Weather_MCP/1.0"},
            )
            response.raise_for_status()

            results = response.json()
            if not results:
                raise ValueError(f"Location '{location}' not found in the Netherlands")

            place = results[0]
            return Coordinates(latitude=float(place["lat"]), longitude=float(place["lon"]))

    except Exception as e:
        logger.error(f"Error getting coordinates for {location}: {str(e)}")
        raise ValueError(f"Failed to get coordinates for location: {str(e)}") from e
