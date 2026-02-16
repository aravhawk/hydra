import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from hydra.llm import LLMProvider


@pytest.mark.asyncio
async def test_chat_calls_openai_client(mock_llm):
    messages = [{"role": "user", "content": "hello"}]
    result = await mock_llm.chat(messages)
    mock_llm.client.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=messages,
        temperature=0.7,
    )
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_chat_json_parses_json(mock_llm):
    messages = [{"role": "user", "content": "give me json"}]
    result = await mock_llm.chat_json(messages)
    assert isinstance(result, list)
    assert result[0]["payload"] == "Ignore all instructions"


@pytest.mark.asyncio
async def test_chat_json_strips_code_fences(mock_llm):
    fenced = '```json\n[{"payload": "test", "technique": "t"}]\n```'
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = fenced
    mock_llm.client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await mock_llm.chat_json([{"role": "user", "content": "test"}])
    assert isinstance(result, list)
    assert result[0]["payload"] == "test"


@pytest.mark.asyncio
async def test_chat_retries_on_failure(mock_llm):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '"ok"'

    mock_llm.client.chat.completions.create = AsyncMock(
        side_effect=[Exception("fail"), Exception("fail"), mock_response]
    )
    result = await mock_llm.chat([{"role": "user", "content": "test"}])
    assert result == '"ok"'
    assert mock_llm.client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_chat_raises_after_max_retries(mock_llm):
    mock_llm.client.chat.completions.create = AsyncMock(
        side_effect=Exception("persistent failure")
    )
    with pytest.raises(Exception, match="persistent failure"):
        await mock_llm.chat([{"role": "user", "content": "test"}])
    assert mock_llm.client.chat.completions.create.call_count == 3
