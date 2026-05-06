from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio


async def get_tools_from_mcp():
    # Define the server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["timezone_server.py"],
    )

    # Connect to the MCP server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
           # Initialize the server connection
           await session.initialize()
           # Ask the server to list all available tools
           response = await session.list_tools()
           print("Connected to MCP server!")
           print("\nAvailable tools:\n")
           for tool in response.tools:
            print(f"- {tool.name}: {tool.description}")
           return response.tools


tools = asyncio.run(get_tools_from_mcp())
print(tools)