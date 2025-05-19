from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent
from mcp import types
from PIL import Image as PILImage
from typing import List
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
from models import AddInput, AddOutput, SqrtInput, SqrtOutput, StringsToIntsInput, StringsToIntsOutput, ExpSumInput, ExpSumOutput, ProductChunkTyped, ProductMetadata, ProductResponse, ProductMetadataSubset
from PIL import Image as PILImage
from tqdm import tqdm
import hashlib
from pathlib import Path
import re
import asyncio

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


from typing import List

# @mcp.tool()
# async def ask_user_for_clarification_feedback(original_user_query: str, query_from_llm: str) -> str:
#     """
#     Prompt the user for clarification or feedback on their original query.
#     """
#     try:
#         print("11")
#         async def get_input():
#             print("\nClarification needed:", query_from_llm)
#             print("Your response: ", end='', flush=True)
#             loop = asyncio.get_event_loop()
#             return await loop.run_in_executor(None, input)
#         print("22")
#         feedback = await asyncio.wait_for(get_input(), timeout=20.0)
#         print("33")
#         print(f"Received feedback: {feedback}")
#         # Return a more structured response that includes both the original query and the feedback
#         return f"{original_user_query} for {feedback.strip()}"
#     except asyncio.TimeoutError:
#         print("Timeout error")
#         return original_user_query


@mcp.tool()
def preety_print_product_metadata_response(product_response_list: List[ProductResponse] | str | List[str]) -> str:
    """
    Pretty print a list of ProductResponse objects' metadata
    Args:
        product_response_list: Either a List[ProductResponse], a JSON string, or a list of JSON strings
    Returns:
        str: Formatted string of product metadata
    """
    try:
        responses = []
        
        # Handle different input types
        if isinstance(product_response_list, str):
            # Single JSON string - Fix escaping before parsing
            try:
                # First try direct JSON parsing
                data = json.loads(product_response_list)
            except json.JSONDecodeError:
                # If that fails, try cleaning the string
                cleaned_json = product_response_list.replace("\\'", "'")  # Replace escaped single quotes
                cleaned_json = cleaned_json.replace('\\"', '"')  # Replace escaped double quotes
                cleaned_json = cleaned_json.replace('\\\\', '\\')  # Replace double backslashes
                data = json.loads(cleaned_json)
                
            if isinstance(data, list):
                responses.extend([ProductResponse(**item) for item in data])
            else:
                responses.append(ProductResponse(**data))
        elif isinstance(product_response_list, list):
            # List of JSON strings or ProductResponse objects
            for item in product_response_list:
                if isinstance(item, str):
                    try:
                        # First try direct JSON parsing
                        data = json.loads(item)
                    except json.JSONDecodeError:
                        # If that fails, try cleaning the string
                        cleaned_json = item.replace("\\'", "'")
                        cleaned_json = cleaned_json.replace('\\"', '"')
                        cleaned_json = cleaned_json.replace('\\\\', '\\')
                        data = json.loads(cleaned_json)
                    responses.append(ProductResponse(**data))
                elif isinstance(item, ProductResponse):
                    responses.append(item)
                else:
                    raise ValueError(f"Invalid item type: {type(item)}")
        else:
            raise ValueError(f"Invalid input type: {type(product_response_list)}")
            
        # Extract and format metadata for each product
        formatted_responses = []
        for product in responses:
            # Access the product_metadata which is a ProductMetadata object
            metadata = product.product_metadata
            metadata_subset = ProductMetadataSubset.from_product_metadata(metadata)
            # Convert metadata to JSON string with indentation
            formatted_responses.append(metadata_subset.model_dump_json(indent=2))
            
        # Join with newlines instead of spaces to better separate products
        return "\n\n".join(formatted_responses)
    except Exception as e:
        return f"Error formatting product response: {str(e)}\nInput: {product_response_list[:200]}..."

@mcp.tool()
def return_ranked_product_response_from_ranked_index(
    product_responses: List[ProductResponse], 
    ranked_indices: List[int]
) -> List[ProductResponse]:
    """
    Reorders a list of ProductResponse objects based on a provided ranking of indices.

    This function is used when a model (e.g., an LLM or retrieval system) provides a ranked list 
    of indices indicating the relevance or order of product responses based on a user query.
    The function maps those indices back to the original list of `ProductResponse` objects and 
    returns them in the ranked order.

    Args:
        product_responses (List[ProductResponse]): The original list of product responses.
        ranked_indices (List[int]): A list of indices representing the new ranked order.

    Returns:
        List[ProductResponse]: A reordered list of product responses based on the ranked indices.

    Example:
        If `product_responses = [A, B, C]` and `ranked_indices = [2, 0]`, 
        the returned result will be `[C, A]`.
    """
    return [product_responses[i] for i in ranked_indices]


@mcp.tool()
def product_metadata_analysis_for_refine_or_tuning_search_result()-> dict:
    """
    Returns a dictionary of unique product attributes and sub-attributes that can be used by an LLM
    to refine, tune, or rerank search results based on user intent.

    This metadata can guide the LLM in determining which fields are important to filter or re-rank 
    products when the `search_product_documents(query: str, top_k: int = 5)` function does not return 
    exact semantic matches.

    For example:
    - In 'master_category', the important field is 'typeName' (e.g., Apparel, Accessories, Footwear).
    - In 'sub_category', the key 'typeName' holds values like Topwear, Bags, Shoes.
    - In 'article_type', 'typeName' includes values such as Tshirts, Backpacks, Water Bottles.
    - For brand, price, ageGroup, gender, season, etc., these fields can help refine or rerank results.

    Example use case:
    If a user searches for "Nike T-shirt for casual wear for men under 100 dollars", the LLM can refine 
    or rerank the results based on:
        - brandName = Nike
        - gender = Men
        - price < 100
        - article_type = T-shirt
        - usage = Casual

    Returns:
        dict: A dictionary with keys representing attribute groups (e.g., article_attributes, metadata)
              and values as lists of relevant attribute names.
    """
    
    article_attributes = ['Add-Ons', 'Ankle Height', 'Arch Type', 'Assorted', 'Back', 'Base Metal', 'Belt Width', 'Blouse', 'Blouse Fabric', 'Body or Garment Size', 'Border', 'Bottom Closure', 'Bottom Fabric', 'Bottom Pattern', 'Bottom Type', 'Brand', 'Brand Fit Name', 'Brick', 'Business Unit', 'Case', 'Character', 'Class', 'Cleats', 'Closure', 'Coin Pocket Type', 'Collar', 'Colour Family', 'Colour Hex Code', 'Colour Shade Name', 'Compartment Closure', 'Concern', 'Content', 'Coverage', 'Cuff', 'Cushioning', 'Design', 'Design Styling', 'Dial Colour', 'Dial Material', 'Dial Pattern', 'Dial Shape', 'Display', 'Distance', 'Distress', 'Dupatta', 'Dupatta Border', 'Dupatta Fabric', 'Dupatta Pattern', 'Effects', 'External Pocket', 'Fabric', 'Fabric 2', 'Fabric 3', 'Fabric Purity', 'Fabric Type', 'Face Shape', 'Fade', 'Family', 'Fastening', 'Fastening and Back Detail', 'Feature', 'Features', 'Features 2', 'Features 3', 'Fine Jewellery', 'Finish', 'Fit', 'Flap Type', 'Fly Type', 'Formulation', 'Fragrance', 'Frame Colour', 'Frame Material', 'Handles', 'Haul Loop Type', 'Heel Height', 'Heel Type', 'Hemline', 'Hood', 'Insole', 'Knit or Woven', 'Laptop Compartment', 'Laptop Size', 'Length', 'Lens Colour', 'Lining', 'Lining Fabric', 'Main Trend', 'Make', 'Material', 'Micro Trend', 'Minimum Shelf Life in Months', 'Minimum Usable Period in Months', 'Model Name', 'Movement', 'Multipack Set', 'Neck', 'Needle', 'Number of Card Holders', 'Number of Compartments', 'Number of Components', 'Number of Contents', 'Number of External Pockets', 'Number of ID Card Holder', 'Number of Inner Pockets', 'Number of Main Compartments', 'Number of Mobile Pouch', 'Number of Panels', 'Number of Pockets', 'Number of Slip Pockets', 'Number of Zips', 'Occasion', 'Ornamentation', 'Outsole Type', 'Padded Shoulder Strap', 'Padding', 'Pattern', 'Pattern Coverage', 'Placket', 'Placket Length', 'Plating', 'Player Type', 'Players', 'Pocket', 'Pocket Type', 'Power Source', 'Print or Pattern Type', 'Processing Time', 'Pronation for Running Shoes', 'Reversible', 'Running Type', 'SPF', 'Saree Fabric', 'Scratch Resistance', 'Seam', 'Segment', 'Set Size', 'Shade', 'Shape', 'Shoe Width', 'Shoulder Strap Type', 'Side Pockets', 'Size', 'Skin Tone', 'Skin Type', 'Sleeve Length', 'Sleeve Styling', 'Sling Strap', 'Slit Detail', 'Sole Material', 'Sport', 'Sport Team', 'Sports Bra Support', 'Stitch', 'Stone Type', 'Strap', 'Strap Closure', 'Strap Colour', 'Strap Material', 'Strap Style', 'Strap Type', 'Straps', 'Strength', 'Stretch', 'Stretchable', 'Style', 'Sub Trend', 'Surface Styling', 'Surface Type', 'Tablet Sleeve', 'Technique', 'Technology', 'Toe Shape', 'Top Design Styling', 'Top Fabric', 'Top Hemline', 'Top Length', 'Top Pattern', 'Top Shape', 'Top Type', 'Total Shelf Life in Months', 'Transparency', 'Trends', 'Type', 'Type of Distress', 'Type of Pleat', 'Units Per Bundle', 'Volume', 'Volume in Litres', 'Waist Rise', 'Waistband', 'Warranty', 'Wash Care', 'Water Resistance', 'Weave Pattern', 'Weave Type', 'Wiring', 'taxMaterial']
    # in master_category typeName is important field is typeName e.g Apparel,Accessories, Footwear, Accessories
    master_category = ['typeName']
    # in sub_category typeName is important field is typeName e.g Topwear, Bags, Shoes, Water Bottle
    sub_category = ['typeName']
    # in article_type typeName is important field is typeName e.g Tshirts, Backpacks, Sports Shoes, Water Bottle
    article_type = ['typeName']
    product_descriptors = ['description', 'materials_care_desc', 'size_fit_desc', 'style_note']
    metadata = ["id","price","discountedPrice","styleType","productTypeId","articleNumber","productDisplayName","variantName","myntraRating",  "catalogAddDate","brandName","ageGroup","gender","baseColour","colour1","colour2","fashionType","season","year","usage","vat","displayCategories"]
    return {
        "article_attributes": article_attributes,
        "master_category": master_category,
        "sub_category": sub_category,
        "article_type": article_type,
        "product_descriptors": product_descriptors,
        "metadata": metadata
    }
    
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

@mcp.tool()
def search_product_documents(query: str, top_k: int = 5)-> list[ProductResponse]:
    """
    Based on the query, search for relevant products from the product documents.
    Return the top_k products.

    @param query: str
    @param top_k: int
    @return list[ProductResponse]
    """
    ensure_faiss_ready()
    mcp_log("SEARCH", f"Query: {query}")
    try:
        index = faiss.read_index(str(ROOT / "faiss_index" / "index.bin"))
        metadata_list = json.loads((ROOT / "faiss_index" / "metadata.json").read_text())
        query_embedding = get_embeddings(query)
        # Reshape query embedding to 2D array as required by faiss
        query_embedding = query_embedding.reshape(1, -1)
        # Search returns (D, I) where D is distances and I is indices
        D, I = index.search(query_embedding, k=top_k)
        print(f"Distances: {D}, Indices: {I}")
        
        results = []
        for idx in I[0]:  # I[0] because I is a 2D array
            if idx < len(metadata_list):  # Ensure index is valid
                data = metadata_list[idx]
                try:
                    # First, properly format the metadata string for JSON parsing
                    metadata_str = data["metadata"]
                    # Replace single quotes with double quotes, but handle nested quotes properly
                    metadata_str = re.sub(r"'(.*?)'", r'"\1"', metadata_str)
                    # Parse the metadata JSON
                    metadata_dict = json.loads(metadata_str)  
                    product_chunk = ProductChunkTyped(
                        id=data["product_id"],
                        product_content=data["chunk"],
                        metadata=ProductMetadata(**metadata_dict)
                    )
                    
                    results.append(ProductResponse.from_product_chunk(product_chunk))
                    # for result in results:
                    #     print(result.model_dump_json(indent=2))
                    #     print("--------------------------------")
                except json.JSONDecodeError as je:
                    mcp_log("ERROR", f"JSON decode error for metadata: {str(je)}\nMetadata string: {metadata_str}")
                    continue
                except Exception as e:
                    mcp_log("ERROR", f"Error processing product chunk: {str(e)}")
                    continue
        
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
        mcp_log("INFO", f"Processing file {i+1} : {file.name}...")
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
            
            new_metadata = {
                "doc": file.name,
                "chunk": product_content,
                "product_id": str(product_id),
                "metadata": json.dumps(metadata.model_dump())
            }
            
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
    mcp_log("INFO", f"ensure_faiss_ready called")
    if not(index_path.exists() and metadata_path.exists()):
        mcp_log("INFO", "Index and metadata file not found - running process_product_documents()...")
        mcp_log("INFO", f"let's called process_product documents called")
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
        print("Indexing documents with MarkItDown...")
        process_product_documents()

        # search_product_documents("backpack")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")



