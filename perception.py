from pydantic import BaseModel
from typing import Optional, List
import os
from dotenv import load_dotenv
from google import genai
import re
from ast import literal_eval

try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class PerceptionResult(BaseModel):
    user_input: str
    modified_user_input: Optional[str] = None
    intent: Optional[str] = None
    entities: List[str] = []
    tool_hint: Optional[str] = None

def extract_perception(user_input: str) -> PerceptionResult:
    """Extracts intent, entities, and tool hints using LLM"""

    prompt = f"""
You are an E-commerce Product Search Agent. Your task is to extract structured information from a user's product-related query.

Input: "{user_input}"

Respond with a Python dictionary containing the following keys:

- intent: A brief phrase describing what the user wants.
- entities: A list of key-value-like strings representing product attributes. 
  For example, if the input is "Nike T-shirt for casual wear for men under age 18", extract:
  ["BrandName:Nike", "ApparelType:T-shirt", "FashionType:Casual", "Gender:Male", "AgeGroup:Under 18"]

- modified_user_input: (Optional) A rewritten version of the query that includes the extracted attributes in natural language.
  Include structured attributes and values if possible. If not applicable, return None.
  Do not include unnecessary words try to include entities or keywords based on the user input.
  For example: "Get a T-shirt with brandName Nike, fashionType Casual, Gender Male, AgeGroup under 18."

- tool_hint: (Optional) Suggest the name of the MCP tool (e.g., "search_product_documents", "ask_user_for_clarification_feedback") that might help, or return None if no tool is needed.

Return only the dictionary in a single line. Do NOT wrap it in ```json or any other formatting. Ensure `entities` is a list of strings, not a dictionary.
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        raw = response.text.strip()
        # Strip ALL Markdown formatting (python, json, etc.)
        clean = re.sub(r"^```\w*\n|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        print("clean", clean, type(clean))
        try:
            parsed = literal_eval(clean)
        except Exception as e:
            log("perception error", f"⚠️ Failed to parse cleaned output: {e}")
            raise
        if isinstance(parsed.get("entities"), dict):
            parsed['entities'] = list(parsed['entities'].values())
        return PerceptionResult(
            user_input=user_input,
            **parsed
        )
    except Exception as e:
        log("ERROR", f"🚨 Error 🔴 extracting perception: {e}")
        return PerceptionResult(user_input=user_input)



if __name__ == "__main__":
    print(extract_perception("Looking for backpack for hiking"))