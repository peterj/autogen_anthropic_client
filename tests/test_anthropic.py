import asyncio
from autogen_anthropic_client import AnthropicChatCompletionClient

def test_anthropic():
    client = AnthropicChatCompletionClient(
        model="claude-3-5-haiku-20241022",
        api_key="apikeyhere"
    )
    
    result = asyncio.run(
        client.create(
            messages=["A haiku about the ocean"],
        )
    )
    print(result.content)

if __name__ == "__main__":
    test_anthropic()