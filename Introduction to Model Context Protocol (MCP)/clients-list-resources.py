import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def list_resources():
    """
    List the available resources from the Timezone Converter MCP server.
    
    """
    # Define server parameters
    params = StdioServerParameters(
        command="python",
        args=["Resources-in-MCP-Servers.py"],
    )
    # Connect to the server
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            # Initialize the server connection
            await session.initialize()

            print(f"Reading resources from MCP server:{resource_uri}")
            resource_content = await session.resource(resource_uri)

            for content in resource_content.contents:
                print(f"\nContent ({content.mimeType}):")
                print(content.text)
            # Ask the server to list all available resources
            response = await session.list_resources()
            print("Connected to MCP server!")
            print("\nAvailable Resources:\n")
            for resource in response.resources:
                print(f"-name: {resource.name}, description: {resource.description}")
            return response.resources


resources = asyncio.run(list_resources())
print(resources)