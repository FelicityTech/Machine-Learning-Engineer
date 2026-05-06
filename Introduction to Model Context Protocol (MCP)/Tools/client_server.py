import asyncio
from mcp.client.stdio import stdio_client
from mcp.session import ClientSession
from mcp.server.stdio import StdioServerParameters

async def get_tools_from_mcp():
    # ...str

    return response.tools

async def call_mcp_tools(tool_name: str, arguments: dict[str, Any]) -> str:
    
    # ...str

    return str(result.content[0].text)

async def call_openai_llm(user_query: str):
    """
    Call OpenAI LLM with MCP tools.
    """

    mcp_tools = await get_tools_from_mcp()

    openai_tools = []
    for tool in mcp_tools:
        openai_tool = {
            "type": "function",
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema, # MCP uses JSON Schema format
        }
        openai_tools.append(openai_tool)



# 1. Send the Query and Tools to LLM
from openai import AsyncOpenAI


async def call_openai_llm(user_query: str):
    # Initialize OpenAI client
    
    client = AsyncOpenAI(api_key="<YOUR_OPENAI_API_KEY>")

    response = await client.responses.create(
        model="gpt-4o-mini",
        input=user_query,
        tools=openai_tools,
    )


# 2. Checking for a Tool Call
async def call_openai_llm(user_query: str):
    # ...

    output = response.output[0]
    if output.type == "function_call":
        args = json.loads(output.arguments)
        name = output.name


        print(f"Model decided to call: {name}")
        print(f"With arguments: {args}\n")


# 3. Calling the Tool | 4. THe Follow - Up Message

async def call_openai_llm(user_query: str):
    # ...

    if output.type == "function_call":
        # ...

        result = await call_mcp_tool(name, args)
        followup = await client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "user", "content": user_query},
                {"type": "function_call_output", "call_id": output.call_id, "output": result},
            ]
        )


# 5. Final Response

async def call_openai_llm(user_query: str):
    # ...
    if output.type == "function_call":
        # ...

        if followup.output and followup.output[0].type == "message":
            print(f"\nAssistant: {followup.output[0].content[0].text}")
        else:
            print("No final response from assistant")
    else:
        print(f"\nAssistant: {output.content[0].text}")
        return str(output.content[0].text)


# Testing the function
if __name__ == "__main__":
    asyncio.run(call_openai_llm("It is 9:50 AM in the UK in January. What time is it in Lisbon, Portugal?"))