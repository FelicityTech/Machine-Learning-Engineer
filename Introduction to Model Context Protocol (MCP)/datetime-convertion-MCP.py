# Step 1: Import necessary libraries
from urllib3 import response
from mcp.server.fastmcp import FastMCP
import datetime
import requests

# create an MCP server instance
mcp=FastMCP("TImezone Converter")

# Step 2: Adding the Tools

@mcp.tool()
def convert_timezone(date_time: str, from_timezone: str, to_timezone: str) -> str:
    """
    Convert date and time from one timezone to another

    Args:
        date_time: The date and time to convert in ISO format.
        from_timezone: The source timezone (e.g., "America/New_York").
        to_timezone: The target timezone (e.g., "Europe/London").

    Returns:
        A string with the converted date and time in the target timezone.
    """
    # Define the API endpoint and format input data
    url = "https://api.opentimezone.com/convert"
    payload={
        "dateTime": date_time,
        "fromTimezone": from_timezone,
        "toTimezone": to_timezone
    }
    
    response = requests.post(url, json=payload)
    data = response.json()
    converted_time = data.get('dateTime', 'N/A')
    return f"Time in {to_timezone}: {converted_time}"


