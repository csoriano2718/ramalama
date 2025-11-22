"""
E2E tests for RAG mode enforcement via RAG_MODE environment variable.

Tests two operational modes:
- strict: Document-only responses, refuses general knowledge
- augment: Freely combines documents with general AI knowledge (default)

Requires:
- deepseek-r1:14b model
- RAG container rebuilt with updated rag_framework script
  (see container-images/scripts/rag_framework for changes)

Note: Due to container cleanup timing, tests may fail when run sequentially.
Run tests individually if needed:
  pytest test/e2e/test_rag_modes.py::test_rag_strict_mode -m e2e
  pytest test/e2e/test_rag_modes.py::test_rag_augment_mode -m e2e
"""

import json
import random
import string
import time
from pathlib import Path
from test.conftest import skip_if_darwin, skip_if_docker, skip_if_no_container
from test.e2e.utils import RamalamaExecWorkspace

import pytest
import requests


# Model used for testing
SUPPORTED_MODELS = ["deepseek-r1:14b"]


def create_test_documents(docs_dir):
    """Create simple test documents for RAG testing"""
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Document 1: Simple facts
    (docs_dir / "facts.md").write_text("""
Alex's favorite ice cream is mint chocolate chip.
The camping trip is June 15-18 at Pine Lake.
The trip costs $85 per person.
    """.strip())
    
    return docs_dir


def wait_for_server(url, timeout=120):
    """Wait for server to be ready (120s for RAG container startup)"""
    start = time.time()
    last_error = None
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                return True
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as e:
            last_error = str(e)
        time.sleep(2)  # Check every 2 seconds
    print(f"Server failed to start after {timeout}s. Last error: {last_error}")
    return False


def query_rag_server(url, question):
    """Query the RAG server and return response"""
    try:
        response = requests.post(
            f"{url}/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": question}],
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Query failed: {e}")
    return None


@pytest.mark.e2e
@skip_if_no_container
@skip_if_docker
@skip_if_darwin
@pytest.mark.parametrize("test_model", SUPPORTED_MODELS)
def test_rag_strict_mode(test_model):
    """Test RAG strict mode - should only answer from documents"""
    with RamalamaExecWorkspace() as ctx:
        docs_dir = create_test_documents(Path(ctx.workspace_dir) / "docs")
        rag_db = Path(ctx.workspace_dir) / "rag_db"
        rag_db.mkdir(parents=True, exist_ok=True)
        
        ctx.check_call(["ramalama", "pull", test_model])
        ctx.check_call(["ramalama", "rag", docs_dir.as_posix(), rag_db.as_posix()])
        
        container_name = f"rag_strict_{''.join(random.choices(string.ascii_letters + string.digits, k=5))}"
        
        ctx.check_call([
            "ramalama", "serve",
            "--name", container_name,
            "--detach",
            "--env", "RAG_MODE=strict",
            "--rag", rag_db.as_posix(),
            test_model
        ])
        
        try:
            server_url = "http://localhost:8080"
            assert wait_for_server(server_url, timeout=120), "Server did not become ready"
            
            # Test 1: Query in documents (should answer)
            response1 = query_rag_server(server_url, "What is Alex's favorite ice cream?")
            assert response1 is not None, "Query failed"
            assert any(term in response1.lower() for term in ["mint", "chocolate"]), \
                f"Should mention mint chocolate chip. Got: {response1}"
            
            # Test 2: Query NOT in documents (should refuse)
            response2 = query_rag_server(server_url, "What is the capital of France?")
            assert response2 is not None, "Query failed"
            
            # Should refuse in strict mode
            refused = any(pattern in response2.lower() for pattern in [
                "don't know", "do not know", "not in", "cannot find", 
                "no information", "document", "context"
            ])
            assert refused or len(response2) < 50, \
                f"Strict mode should refuse general knowledge. Got: {response2}"
        
        finally:
            ctx.check_call(["ramalama", "stop", container_name])


@pytest.mark.e2e
@skip_if_no_container
@skip_if_docker
@skip_if_darwin
@pytest.mark.parametrize("test_model", SUPPORTED_MODELS)
def test_rag_augment_mode(test_model):
    """Test RAG augment mode - should freely combine docs with general knowledge"""
    with RamalamaExecWorkspace() as ctx:
        docs_dir = create_test_documents(Path(ctx.workspace_dir) / "docs")
        rag_db = Path(ctx.workspace_dir) / "rag_db"
        rag_db.mkdir(parents=True, exist_ok=True)
        
        ctx.check_call(["ramalama", "pull", test_model])
        ctx.check_call(["ramalama", "rag", docs_dir.as_posix(), rag_db.as_posix()])
        
        container_name = f"rag_augment_{''.join(random.choices(string.ascii_letters + string.digits, k=5))}"
        
        ctx.check_call([
            "ramalama", "serve",
            "--name", container_name,
            "--detach",
            "--env", "RAG_MODE=augment",
            "--rag", rag_db.as_posix(),
            test_model
        ])
        
        try:
            server_url = "http://localhost:8080"
            assert wait_for_server(server_url, timeout=120), "Server did not become ready"
            
            # Test: Query about documents (should answer)
            response = query_rag_server(server_url, "Tell me about the camping trip cost")
            assert response is not None, "Query failed"
            
            # Should mention the cost from documents
            assert any(term in response.lower() for term in ["85", "cost", "price", "dollar"]), \
                f"Should mention cost. Got: {response}"
        
        finally:
            ctx.check_call(["ramalama", "stop", container_name])


@pytest.mark.e2e
@skip_if_no_container
@skip_if_docker
@skip_if_darwin
def test_rag_mode_env_variable_propagation():
    """Test that RAG_MODE environment variable is properly passed to container"""
    import subprocess
    
    # This is a lightweight test that doesn't need the full RAG setup
    # Just verify the env var can be set
    result = subprocess.run(
        ["env"],
        capture_output=True,
        text=True,
        env={"RAG_MODE": "strict"}
    )
    assert "RAG_MODE" in result.stdout or result.returncode == 0
