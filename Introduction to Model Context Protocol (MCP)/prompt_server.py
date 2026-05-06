# Define a prompt for currency conversion
@mcp.prompt(title="Currency Conversion")
def list_prompt(currency_request: str) -> str:
    return f"""You are a currency conversion assistant.

Your task is to:
1. Extract the amount and source currency from the user's natural language input.
2. Identify the target currency.
3. Use the conversion tool to convert the amount.

Rules:
- If the amount or currencies are ambiguous or missing, ask the user for clarification.
- Use only supported currency codes (e.g., USD, EUR, GBP).

User's currency conversion request: {currency_request}"""

# Test the prompt function
print(list_prompt("100 USD to EUR"))