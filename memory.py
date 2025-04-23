import faiss
import numpy as np
import requests
from typing import List, Optional, Literal
from pydantic import BaseModel
from datetime import datetime


class MemoryRecord(BaseModel):
    text: str
    type: Literal["preference", "tool_output", "fact", "query", "system"] = "fact"
    timestamp: Optional[str] = datetime.now().isoformat()
    tool_name: Optional[str] = None
    user_query: Optional[str] = None
    tags: List[str] = []
    session_id: Optional[str] = None

EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"



class MemoryManager:
    """
    A memory management utility that stores and retrieves text-based memory records 
    using vector embeddings and FAISS for similarity search.

    Attributes:
        embedding_model_url (str): URL of the embedding model inference endpoint.
        model_name (str): Name of the embedding model to be used.
        index (faiss.Index): FAISS index for similarity search.
        data (List[MemoryRecord]): List of stored memory records.
        embeddings (List[np.ndarray]): List of corresponding vector embeddings.
    """

    def __init__(self, embedding_model_url=EMBED_URL, model_name=EMBED_MODEL):
        """
        Initializes the MemoryManager with the given embedding model settings.
        
        Args:
            embedding_model_url (str): The URL to the embedding model API.
            model_name (str): The name of the embedding model to use.
        """
        self.embedding_model_url = embedding_model_url
        self.model_name = model_name
        self.index = None
        self.data: List[MemoryRecord] = []
        self.embeddings: List[np.ndarray] = []

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Gets the embedding vector for the given text from the remote model.

        Args:
            text (str): The input text to embed.

        Returns:
            np.ndarray: The embedding vector as a NumPy array.
        """
        response = requests.post(
            self.embedding_model_url,
            json={"model": self.model_name, "prompt": text},
        )
        response.raise_for_status()
        return np.array(response.json()["embedding"], dtype=np.float32)

    def add(self, item: MemoryRecord):
        """
        Adds a single memory record and its embedding to the FAISS index.

        Args:
            item (MemoryRecord): The memory record to add.
        """
        emb = self._get_embedding(item.text)
        self.embeddings.append(emb)
        self.data.append(item)

        if self.index is None:
            self.index = faiss.IndexFlatL2(len(emb))
        self.index.add(np.stack([emb]))
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        type_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
        session_filter: Optional[str] = None
    ) -> List[MemoryRecord]:
        """
        Retrieves the top_k most similar memory records to the given query, 
        optionally applying filters.

        Args:
            query (str): The input query string.
            top_k (int): The number of results to return.
            type_filter (Optional[str]): Filter by record type.
            tag_filter (Optional[List[str]]): Filter by tags.
            session_filter (Optional[str]): Filter by session ID.

        Returns:
            List[MemoryRecord]: A list of the most relevant memory records.
        """
        if self.index is None or len(self.data) == 0:
            return []
        
        query_vec = self._get_embedding(query).reshape(1, -1)
        D, I = self.index.search(query_vec, top_k*2) # Overfetch to allow filtering

        results = []
        for idx in I[0]:
            if idx >= len(self.data):
                continue
            item = self.data[idx]

            # if type filter
            if type_filter and item.type != type_filter:
                continue
            
            # if tag filter
            if tag_filter and not any(tag in item.tags for tag in tag_filter):
                continue
            # if session filter
            if session_filter and item.session_id != session_filter:
                continue

            results.append(item)
            if len(results) >= top_k:
                break

        return results

    def bulk_add(self, items: List[MemoryRecord]):
        """
        Adds multiple memory records to the memory store in bulk.

        Args:
            items (List[MemoryRecord]): A list of memory records to add.
        """
        for item in items:
            self.add(item)
            
            
            
        
        
