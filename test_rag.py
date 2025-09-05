import pytest
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8009"

class TestRAGAPI:
    
    def test_health_endpoint(self):
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_query_endpoint(self):
        query_data = {"query": "What is the main topic?"}
        response = requests.post(f"{BASE_URL}/query", json=query_data)
        assert response.status_code in [200, 429]  # 429 for rate limit
        
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "processing_time" in data
    
    def test_metrics_endpoint(self):
        response = requests.get(f"{BASE_URL}/metrics")
        assert response.status_code in [200, 429]
        
        if response.status_code == 200:
            data = response.json()
            assert "documents_indexed" in data

def test_document_processor():
    from document_processor import DocumentProcessor
    
    # Test text processing
    test_file = Path("test.txt")
    test_file.write_text("Test content")
    
    result = DocumentProcessor.process_document(test_file)
    assert result["content"] == "Test content"
    assert result["metadata"]["type"] == "text"
    
    test_file.unlink()  # cleanup

if __name__ == "__main__":
    pytest.main([__file__, "-v"])