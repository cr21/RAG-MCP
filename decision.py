from perception import PerceptionResult
from memory import MemoryRecord
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
import os


try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_plan(
    perception: PerceptionResult,
    memory_items: List[MemoryRecord],
    tool_descriptions: Optional[str] = None
) -> str:
    """
    Generates a plan (tool call or final answer) using LLM based on structured perception and memory.

    Args:
        perception (PerceptionResult): The structured perception result.
        memory_items (List[MemoryRecord]): The list of memory items.
        tool_descriptions (Optional[str]): The description of the available tools.

    Returns:
        str: The plan (tool call or final answer).
    """

    memory_texts = "\n".join(f"- {m.text}" for m in memory_items) or "None"
    
    tool_context = f"\nYou have access to the following tools:\n{tool_descriptions}" if tool_descriptions else ""

    

    prompt = f"""
You are Ecommerce Product Finder/ Search Agent. Your job is to recommend products  or help user find right product based on their search query. You can use tools to search for products If need, and continue until the FINAL_ANSWER is produced.{tool_context}

Always follow this loop:


1. Think step-by-step about the problem.
2. If the user query is unclear, ask for clarification concisely, focusing only on product attributes, entities, or metadata.
3. If a tool is needed, respond using the format:
   FUNCTION_CALL: tool_name|param1=value1|param2=value2
4. When the retrieval result is known, refine or rerank the results based on relevance to the user query. Use tools like `product_metadata_analysis_for_refine_or_tuning_search_result` if needed.
5. When the final answer is ready, respond using:
   FINAL_ANSWER: [your final result]


Guidelines:
- Respond using EXACTLY ONE of the formats above per step.
- If user query is not clear, or need refinement,modify user query by asking some feedback questions, but be very consice and modify only if needed, modify interms of product attributes, entities, brands or some kind of metadata and values, no verbose modification.
- Do NOT include extra text, explanation, or formatting.
- Use nested keys (e.g., input.string) and square brackets for lists.
- You can reference these relevant memories:
{memory_texts}

Input Summary:
- User input: "{perception.user_input}"
- Intent: {perception.intent}
- Entities: {', '.join(perception.entities)}
- Tool hint: {perception.tool_hint or 'None'}

✅ Examples:
- FUNCTION_CALL: ask_user_for_clarification_feedback|original_user_query="Find Nike T-shirt for Men",query_from_llm="Is this T-shirt for Casual wear or sports"
- FUNCTION_CALL: return_ranked_product_response_from_ranked_index|product_responses=[ProductResponse(name="Nike T-shirt for Men", price=100, description="This is a Nike T-shirt for Men"), ProductResponse(name="Adidas T-shirt for Men", price=80, description="This is an Adidas T-shirt for Men")],ranked_indices=[0,1]
- FUNCTION_CALL: product_metadata_analysis_for_refine_or_tuning_search_result
- FUNCTION_CALL: search_product_documents|query="Find Nike T-shirt for Men",top_k=5
- FUNCTION_CALL: int_list_to_exponential_sum|input.int_list=[73,78,68,73,65]
- FINAL_ANSWER: [ProductResponse]

✅ Examples:
- User asks: "Nike T-shirt for casual wear"
- FUNCTION_CALL: ask_user_for_clarification_feedback|original_user_query="Find Nike T-shirt for Men",query_from_llm="Is this T-shirt for Casual wear or sports"
  - [Recieves feedback from user or modified query]
  - FUNCTION_CALL: search_product_documents|query=modified_query,top_k=5
  - [Receives a list of product responses]
  - FUNCTION_CALL: product_metadata_analysis_for_refine_or_tuning_search_result
  - [Received detail metadata of product]
  - Idetify the Ranked indices of product responses based on relevance to user query
  - FUNCTION_CALL: return_ranked_product_response_from_ranked_index|product_responses=[ProductResponse(name="Nike T-shirt for Men", price=100, description="This is a Nike T-shirt for Men"), ProductResponse(name="Adidas T-shirt for Men", price=80, description="This is an Adidas T-shirt for Men")],ranked_indices=[0,1]
  - [receives a final answer]
  - FINAL_ANSWER: [ProductResponse]


IMPORTANT:
- 🚫 Do NOT invent tools. Use only the tools listed below.
- If You need to modify or extract some information from the product metadata, you can use tools to get plan how to do that e.g product_metadata_analysis_for_refine_or_tuning_search_result
- 🤖 If the previous tool output already contains factual information, DO NOT search again. Instead, summarize the relevant facts based on metadata matching or analysisand respond with: FINAL_ANSWER: [your answer]
- Only repeat `search_product_documents` if the last result was irrelevant or empty, or never call search_product_documents with same input query.
- ❌ Do NOT repeat function calls with the same parameters.
- ❌ Do NOT output unstructured responses.
- 🧠 Think before each step. Verify intermediate results mentally before proceeding.
- 💥 If unsure or no tool fits, skip to FINAL_ANSWER: [unknown]
- ✅ You have only 3 attempts. Final attempt must be FINAL_ANSWER]
"""
    

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        raw = response.text.strip()
        log("plan", f"LLM output: {raw}")

        for line in raw.splitlines():
            if line.strip().startswith("FUNCTION_CALL:") or line.strip().startswith("FINAL_ANSWER:"):
                return line.strip()
        return raw.strip()
    except Exception as e:
        log("plan", f"⚠️ Decision generation failed: {e}")
        return "FINAL_ANSWER: [unknown]"
