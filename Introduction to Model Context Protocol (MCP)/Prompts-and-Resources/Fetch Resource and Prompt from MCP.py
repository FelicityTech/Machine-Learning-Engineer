from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def get_context_from_mcp(user_query: str) -> tuple[str, str]:
    """Fetch resource content and prompt text from the MCP server."""
    params = StdioServerParameters(command=sys.executable, args=["currency_server.py"])

    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            # Read the resource (supported currencies)
            resource_result = await session.read_resource("file://currencies.txt")
            resource_text = resource_result.contents[0].text

            # Get the prompt with the user's query
            prompt_result = await session.get_prompt("convert_currency_prompt",
                arguments={"currency_request": user_query})
            prompt_text = prompt_result.messages[0].content.text

            return resource_text, prompt_text

print(asyncio.run(get_context_from_mcp("How much is 50 GBP in euros?")))