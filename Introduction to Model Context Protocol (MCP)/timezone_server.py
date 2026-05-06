import mcp
from mcp.server.fastmcp import FastMCP
import requests

# Create an MCP server instance
mcp = FastMCP("Timezone Converter")

@mcp.tool()
def convert_timezone(date_time: str, from_timezone: str, to_timezone: str)->str:
    """
    Convert date and time from one timezone to another

    Args:
        date_time: The date and time to convert in ISO format.
        from_timezone: The source timezone (e.g., "America/New_York").
        to_timezone: The target timezone (e.g., "Europe/London").

    Returns:
        A string with the converted date and time in the target timezone.
    """
    url = "https://api.opentimezone.com/convert"
    payload = {
        "dateTime": date_time,
        "fromTimezone": from_timezone,
        "toTimezone": to_timezone,
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
        return str(data)
    except Exception:
        # Fallback: use pytz for local conversion if the API is unavailable
        from datetime import datetime
        import pytz
        src_tz = pytz.timezone(from_timezone)
        dst_tz = pytz.timezone(to_timezone)
        dt = src_tz.localize(datetime.fromisoformat(date_time))
        converted = dt.astimezone(dst_tz)
        return converted.strftime("%Y-%m-%dT%H:%M:%S %Z")

if __name__ == "__main__":
    mcp.run(transport="stdio")