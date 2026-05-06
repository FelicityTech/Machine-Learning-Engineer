from click import argument
import async

async def read_prompt(user_input:str, prompt_name:str = "convert_timezone_prompt") -> str:
    """Read a specific prompt from the MCP server."""
    params = StdioServerParameters(
        command="python",
        args=["timezone_server1.py"],
    )
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

    prompts = await session.list_prompts()
    # Retrieve  the prompt
    if prompts.prompts:
        prompt = await session.get_prompt(prompt_name, arguments={"timezone_request": user_input})
        print(f"Prompt result:{prompt.message[0].content.text}")
    

    return prompt.message[0].content.text

# Example usage:
prompt_uri = "file://timezone_prompt.txt"
prompt = asyncio.run(read_prompt(user_input="It is 9:50 AM in the UK in January. What time is it in Lisbon, Portugal?"))
print(prompt)