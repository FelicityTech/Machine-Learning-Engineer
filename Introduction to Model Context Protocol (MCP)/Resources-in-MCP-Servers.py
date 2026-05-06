# Defining MCP server Resources
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Timezone COnverter")


@mcp.resource("file://locations.txt")
def get_locations() -> str: 
    """
    Get the list of cities for timezone conversion

    Returns:
        Comtent of the locations.txt file with coty names
    """
    try:
        with open("locations.txt", "r") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return "No locations found. The file locations.txt is missing."
        

if __name__ == "__main__":
    mcp.run(transport="stdio")