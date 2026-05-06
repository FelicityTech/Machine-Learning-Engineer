import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def call_mcp_tool(tool_name:str, arguments: dict) -> str:
    params = StdioServerParameters(command=sys.executable, args=["timezone_server.py"])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:

            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            text_content = result.content[0].text
            print(f"Conversion Result: {text_content}")
            return str(text_content)
            

asyncio.run(
    call_mcp_tool(
        tool_name="convert_timezone",
        arguments={
            "date_time": "2026-05-06T10:00:00",
            "from_timezone": "America/New_York",
            "to_timezone": "Europe/London"
        }
    )
)