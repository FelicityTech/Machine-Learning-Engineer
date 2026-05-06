from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Timezone Converter")

@mcp.tool()
def convert_timezone_prompt(timezone_request: str) -> str:
    """
    """
    


if __name__ == "__main__":
    mcp.run(transport="stdio")
