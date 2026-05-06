from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Timezone Converter")


@mcp.tool()
def convert_timezone(date_time:str, from_timezone: str, to_timezone:str) -> str:
    """Convert a date and time from one timezone to another"""



if __name__ == "__main__":
    mcp.run(transport="stdio")