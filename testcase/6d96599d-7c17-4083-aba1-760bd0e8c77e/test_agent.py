
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from starlette.testclient import TestClient
import agent

@pytest.fixture
def client():
    """Fixture to provide a FastAPI test client."""
    with TestClient(agent.app) as c:
        yield c

@pytest.mark.asyncio
async def test_functional_valid_query_returns_answer(monkeypatch):
    """
    Validates that a well-formed user query about finance policy returns a successful answer via the /query endpoint.
    Mocks all external dependencies (Azure AI Search, OpenAI API) 
    """
    # Patch ChunkRetriever.retrieve_chunks to return mock context chunks
    mock_chunks = ["Travel expenses are reimbursed according to section 4.2 of the policy."]
    with patch.object(agent.ChunkRetriever, "retrieve_chunks", new_callable=AsyncMock) as mock_retrieve_chunks, \
         patch.object(agent.LLMService, "generate_response", new_callable=AsyncMock) as mock_generate_response:
        mock_retrieve_chunks.return_value = mock_chunks
        mock_generate_response.return_value = (
            "The reimbursement policy for travel expenses is outlined in section 4.2 of the XYZ Finance Policies. "
            "Employees must submit receipts and obtain manager approval."
        )

        with TestClient(agent.app) as client:
            response = client.post(
                "/query",
                json={"user_query": "What is the reimbursement policy for travel expenses?"}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["answer"] is not None
            assert data["error"] is None
            assert data["tips"] is None
            # Optionally, check that the answer contains expected keywords
            assert "reimbursement policy" in data["answer"].lower()
            assert "travel expenses" in data["answer"].lower()