from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Currency Converter")

# Define a resource for the currencies file
@mcp.resource("file://currencies.txt")
def get_currencies() -> str:
    """
    Get the list of currency names published by the European Central Bank for currency conversion.

    Returns:
        Contents of the currencies.txt file with currency names
    """
    # Open currencies.txt and read the data
    try:
        with open('currencies.txt', 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return "currencies.txt file not found"

# Test the resource function
print(get_currencies())