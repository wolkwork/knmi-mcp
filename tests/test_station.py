import pytest
from knmi_weather_mcp.station import StationManager
from knmi_weather_mcp.models import WeatherStation, Coordinates

@pytest.mark.asyncio
async def test_station_manager_initialization():
    manager = StationManager()
    assert isinstance(manager.stations, dict)

@pytest.mark.asyncio
async def test_station_refresh():
    manager = StationManager()
    await manager.refresh_stations()
    assert len(manager.stations) > 0
    
    # Verify station data structure
    for station in manager.stations.values():
        assert isinstance(station, WeatherStation)
        assert isinstance(station.coordinates, Coordinates)
        assert -90 <= station.coordinates.latitude <= 90
        assert -180 <= station.coordinates.longitude <= 180