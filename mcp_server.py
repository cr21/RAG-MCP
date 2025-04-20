from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent
from mcp import types
from PIL import Image as PILImage
import math
import sys
import os
import json
import faiss
import numpy as np
from pathlib import Path
import requests
from markitdown import MarkItDown
import time
import logging
from models import AddInput, AddOutput, SqrtInput, SqrtOutput, StringsToIntsInput, StringsToIntsOutput, ExpSumInput, ExpSumOutput, ProductChunkTyped, ProductMetadata
from PIL import Image as PILImage
from tqdm import tqdm
import hashlib
from pathlib import Path

mcp = FastMCP("Agent")

## OLLAMA Embedding Model
EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
ROOT = Path(__file__).parent.resolve()

def get_embeddings(text: str)-> np.ndarray:
    """
    Get the embeddings for a text using the  Embedding Model
    """
    response = requests.post(
        url=EMBED_URL,
        json={
            "model": EMBED_MODEL,
            "prompt": text
        }
    )
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)
    
def mcp_log(level: str, message: str) -> None:
    """
    Log a message to stderr to avoid interfering with JSON communication
    """
    sys.stderr.write(f"{level}: {message}\n")
    sys.stderr.flush()
    
@mcp.tool()
def add(input: AddInput) -> AddOutput:
    """
    Add two numbers
    """
    print("CALLED: add(AddInput) -> AddOutput")
    return AddOutput(result=input.a + input.b)

@mcp.tool()
def sqrt(input: SqrtInput) -> SqrtOutput:
    """
    Square root of a number
    """
    print("CALLED: sqrt(SqrtInput) -> SqrtOutput")
    return SqrtOutput(result=math.sqrt(input.a))

@mcp.tool()
def subtract(  a:int, b: int) -> int:    
    """
    Subtract two numbers
    """
    print("CALLED: subtract(a: int, b: int) -> int")
    return int(a - b)

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers
    """
    print("CALLED: multiply(a: int, b: int) -> int")
    return int(a * b)

@mcp.tool()
def divide(a: int, b: int) -> float:
    """
    Divide two numbers
    """
    print("CALLED: divide(a: int, b: int) -> float")
    return float(a / b)

@mcp.tool()
def power(a: int, b: int) -> int:
    """
    Power of two numbers
    """
    print("CALLED: power(a: int, b: int) -> int")
    return int(a ** b)

@mcp.tool()
def cbrt(a: int) -> float:
    """
    Cube root of a number
    """
    print("CALLED: cbrt(a: int) -> float")
    return float(a ** (1/3))

@mcp.tool()
def factorial(a: int) -> int:           
    """
    Factorial of a number
    """
    print("CALLED: factorial(a: int) -> int")
    return int(math.factorial(a))

@mcp.tool()
def log(a: int) -> float:
    """
    Log of a number
    """
    print("CALLED: log(a: int) -> float")
    return float(math.log(a))

@mcp.tool()
def remainder(a: int, b: int) -> int:
    """
    Remainder of two numbers
    """
    print("CALLED: remainder(a: int, b: int) -> int")
    return int(a % b)

@mcp.tool()
def sin(a: int) -> float:
    """
    Sine of a number
    """
    print("CALLED: sin(a: int) -> float")
    return float(math.sin(a))

@mcp.tool()
def cos(a: int) -> float:
    """
    Cosine of a number
    """
    print("CALLED: cos(a: int) -> float")
    return float(math.cos(a))

@mcp.tool()
def tan(a: int) -> float:
    """
    Tangent of a number
    """
    print("CALLED: tan(a: int) -> float")
    return float(math.tan(a))

@mcp.tool()
def create_thumbnail(image_path: str) -> Image:
    """Create a thumbnail from an image"""
    print("CALLED: create_thumbnail(image_path: str) -> Image:")
    img = PILImage.open(image_path)
    img.thumbnail((100, 100))
    return Image(data=img.tobytes(), format="png")

@mcp.tool()
def strings_to_chars_to_int(input: StringsToIntsInput) -> StringsToIntsOutput:
    """Return the ASCII values of the characters in a word"""
    print("CALLED: strings_to_chars_to_int(StringsToIntsInput) -> StringsToIntsOutput")
    ascii_values = [ord(char) for char in input.string]
    return StringsToIntsOutput(ascii_values=ascii_values)

@mcp.tool()
def int_list_to_exponential_sum(input: ExpSumInput) -> ExpSumOutput:
    """Return sum of exponentials of numbers in a list"""
    print("CALLED: int_list_to_exponential_sum(ExpSumInput) -> ExpSumOutput")
    result = sum(math.exp(i) for i in input.int_list)
    return ExpSumOutput(result=result)

@mcp.tool()
def fibonacci_numbers(n: int) -> list:
    """Return the first n Fibonacci Numbers"""
    print("CALLED: fibonacci_numbers(n: int) -> list:")
    if n <= 0:
        return []
    fib_sequence = [0, 1]
    for _ in range(2, n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence[:n]



# DEFINE AVAILABLE PROMPTS
@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"
    print("CALLED: review_code(code: str) -> str:")


@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]

def search_product_documents(query: str)-> list[ProductChunkTyped]:
    """
    Search relevant content from uploaded products.
    """
    ensure_faiss_ready()
    mcp_log("SEARCH", f"Query: {query}")
    try:
        index = faiss.read_index(str(ROOT / "faiss_index" / "index.bin"))
        metadata_list = json.loads((ROOT / "faiss_index" / "metadata.json").read_text())
        query_embedding = get_embeddings(query)
        print(f"query_embedding: {query_embedding}")
        # Reshape query embedding to 2D array as required by faiss
        query_embedding = query_embedding.reshape(1, -1)
        # Search returns (D, I) where D is distances and I is indices
        D, I = index.search(query_embedding, k=5)
        print(f"Distances: {D}, Indices: {I}")
        
        results = []
        for idx in I[0]:  # I[0] because I is a 2D array
            if idx < len(metadata_list):  # Ensure index is valid
                data = metadata_list[idx]
                print(f"data: {data}")
                results.append(ProductChunkTyped(
                    id=data["product_id"],
                    product_content=data["chunk"],
                    metadata=ProductMetadata(**json.loads(data["metadata"].replace("'", '"')))
                ))
        for result in results:
            print(result.model_dump_json(indent=2))
            
            print("--------------------------------")
        return results
            
    except Exception as e:
        mcp_log("ERROR", f"Failed to search product documents: {e}")
        return []
    
      
def process_product_documents():
    """
    Process the product documents, Parse in proper format compiled into pydantic model ProductChunkTyped,
    Index using Faiss and save to disk.
    Maintain a hash of the processed documents to avoid reprocessing unless necessary.
    Maintain a hash of the index to avoid reindexing unless necessary.
    Maintain metadata 
    """
    mcp_log("INFO", "Indexing documents with MarkItDown...")
    ROOT = Path(__file__).parent.resolve()
    DOC_PATH = ROOT / "documents"
    INDEX_CACHE = ROOT / "faiss_index"
    INDEX_CACHE.mkdir(exist_ok=True)
    INDEX_FILE = INDEX_CACHE / "index.bin"
    METADATA_FILE = INDEX_CACHE / "metadata.json"
    CACHE_FILE = INDEX_CACHE / "product_index_cache.json"

    def file_md5_hash(path):
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    
    CACHE_META = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    metadata_list = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else []
    index = faiss.read_index(str(INDEX_FILE)) if INDEX_FILE.exists() else None
    # all_embeddings = []
    converter= MarkItDown()
    
    for i, file in enumerate(DOC_PATH.glob("*.json")):
        print(f"Processing file {i+1} : {file.name}...")
        f_md_hash = file_md5_hash(file)
        if file.name in CACHE_META and CACHE_META[file.name] == f_md_hash:
            mcp_log("INFO", f"Skipping {file.name} - already processed.")
            continue
        try:
            mcp_log("INFO", f"Processing {file.name}...")
            # file is not processed, process it
            product_data = json.load(open(file))
            product = ProductChunkTyped.from_json(product_data)
            product_content = product.product_content
            metadata = product.metadata
            product_id = product.id
            embedding = get_embeddings(product_content)
            #all_embeddings.append(embedding)
            new_metadata = {"doc": file.name, "chunk": product_content, "product_id": f"{product_id}","metadata": f"{metadata.model_dump_jsonC()}"}
            
            metadata_list.append(new_metadata)
            if embedding is not None:
                if index is None:
                    dim = len(embedding)
                    index = faiss.IndexFlatL2(dim)
                index.add(np.stack([embedding]))
                CACHE_META[file.name] = f_md_hash
                   
        except Exception as e:
            mcp_log("ERROR", f"Failed to process {file.name}: {e}")
        
    CACHE_FILE.write_text(json.dumps(CACHE_META, indent=2))
    METADATA_FILE.write_text(json.dumps(metadata_list, indent=2))
    if index and index.ntotal > 0:
        faiss.write_index(index, str(INDEX_FILE))
        mcp_log("SUCCESS", "Saved FAISS index and metadata")
    else:
        mcp_log("WARN", "No new documents or updates to process.")
        

def ensure_faiss_ready():
    """
    Ensure that the Faiss index is ready.
    """
    index_path = ROOT / "faiss_index" / "index.bin"
    metadata_path = ROOT / "faiss_index" / "metadata.json"

    if not(index_path.exists() and metadata_path.exists()):
        mcp_log("INFO", "Index and metadata file not found - running process_product_documents()...")
        process_product_documents()
    else:
        mcp_log("INFO", "Index already exists. Skipping regeneration.")



if __name__ == "__main__":
    print("START MCP SERVER")
    
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        mcp.run() # Run without transport for dev server
    else:
        # Start the server in a separate thread
        import threading
        server_thread = threading.Thread(target=lambda: mcp.run(transport="stdio"))
        server_thread.daemon = True
        server_thread.start()   
    
        time.sleep(2)
        # process_product_documents()
        search_product_documents("Nike Polo T-shirt")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")



