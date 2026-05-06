# Fetch Resource and Prompt from MCP server
from First MCP Server import result
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def get_context_from_mcp(user_query: str):
    """Fetch resource content and prompt from MCP server."""
    # MCP Server connection
    params = StdioServerParameters(
        command=sys.executable,
        args=["timezone_server.py"],
    )

    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            # Get the resource (supported resource URI"file://timezone.json")
            resource = await session.read_resource("file://timezone.json")
            resource_text = resource_result.contents[0].text
            
            # Get the prompt with the user's query
            prompt_result = await session.get_prompt("timezone_prompt_template", arguments={"timezone_request": user_query})
            prompt_text = prompt_result.messages[0].content.text
            return resource_text, prompt_text


# Build the System Message and Call the LLM
async def call_llm_with_context(user_query: str):
    """Call the LLM with resource and prompt context from MCP."""
    resource_text, prompt_text = await get_context_from_mcp(user_query)
    # Combine prompt (task + rules + user request) with resource (supported locations)
    full_prompt = prompt_text + "\nSupported locations:\n" + "Resource: " + resource_text
    client = AsyncOpenAI(api_key="<OPENAI_API_KEY>") 
    response = await client.responses.create(
    model="gpt-4o-mini",
    input=full_prompt,
    tools=openai_tools, # from get_tools_from_mcp(), formatted for OpenAI API

   )
    return response.output[0].content[0].text    
               
        

# 3. Hnadling the Response from the LLM
output = response.output[0]
if output.type == "message":
    print(f"\nAssistant: {output.content[0].text}")
    
else:
    print(f"\nOutput Type: {output.type}")
    print(f"{output}")

    if output.type == "function_call":
        args = json.loads(output.arguments)
        result = await call_mcp_tools(output.name, args)
        followup = await client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "user", "content": user_query},
                output,
                {"type": "function_call_output", "call_id": output.call_id, "output": result},
            ]
        )

        # .. then print the assistant final message
        if followup.output and followup.output[0].type == "message":
            print(f"\nAssistant: {followup.output[0].content[0].text}")
        else:
            print("No final response from assistant")
    else:
        print(f"\nAssistant: {output.content[0].text}")
        return str(output.content[0].text)

        # Testing the function
        if __name__ == "__main__":
            asyncio.run(call_llm_with_context("It is 9:50 AM in the UK in January. What time is it in Lisbon, Portugal?"))    