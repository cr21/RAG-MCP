from typing import Dict, Any, Union
from pydantic import BaseModel
from mcp import ClientSession
import ast
import asyncio
import sys
import datetime
# # Optional: import log from agent if shared, else define locally
# try:
#     from agent import log
# except ImportError:
#     import datetime
#     def log(stage: str, msg: str):
#         now = datetime.datetime.now().strftime("%H:%M:%S")
#         print(f"[{now}] [{stage}] {msg}")

def log(stage: str, msg: str):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{stage}] {msg}")
class ToolCallResult(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    result: Union[str, list, dict]
    raw_response: Any

def parse_function_call(response: str) -> tuple[str, Dict[str, Any]]:
    """Parses FUNCTION_CALL string into tool name and arguments."""
    try:
        print("response parse_function_call", response)
        if not response.startswith("FUNCTION_CALL:"):
            raise ValueError("Not a valid FUNCTION_CALL")

        _, function_info = response.split(":", 1)
        print("function_info", function_info)
        parts = [p.strip() for p in function_info.split("|")]
        if len(parts) == 1:
            func_name = parts[0]
            param_parts = []
        else:
            func_name, param_parts = parts[0], parts[1:]

        result = {}
        for part in param_parts:
            if "=" not in part:
                raise ValueError(f"Invalid param: {part}")
            key, value = part.split("=", 1)

            try:
                parsed_value = ast.literal_eval(value)
            except Exception:
                parsed_value = value.strip()

            # Handle nested keys
            keys = key.split(".")
            current = result
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = parsed_value

        log("parser", f"Parsed: {func_name} → {result}")
        return func_name, result

    except Exception as e:
        log("parser", f"❌ Failed to parse FUNCTION_CALL: {e}")
        raise



async def execute_tool(session: ClientSession, tools: list[Any], response: str) -> ToolCallResult:
    """
    Executes a FUNCTION_CALL via MCP tool session.
    """
    try:
        tool_name, args = parse_function_call(response)
        tool = next((t for t in tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registered tools")
        
        log("tool", f"⚙️ Calling '{tool_name}' with: {args}")
        
        # # Special handling for user input tool
        # if tool_name == "ask_user_for_clarification_feedback":
        #     print("1")
        #     # Ensure all output is flushed before waiting for input
        #     sys.stdout.flush()
        #     sys.stderr.flush()
            
        #     # Call the tool and wait for result
        #     result = await session.call_tool(tool_name, arguments=args)
        #     print("2", result)
        #     # Better result handling
        #     if isinstance(result, str):
        #         out = result
        #     elif hasattr(result, 'content') and isinstance(result.content, list):
        #         # Extract text from content list
        #         out = ' '.join(item.text for item in result.content if hasattr(item, 'text'))
        #     else:
        #         # Fallback to string representation
        #         out = str(result)
            
        #     log("tool", f"Received user feedback: {out}")
        #     return ToolCallResult(
        #         tool_name=tool_name,
        #         arguments=args,
        #         result=out,
        #         raw_response=result
        #     )
        
        # Normal tool execution
        result = await session.call_tool(tool_name, arguments=args)
        
        if hasattr(result, 'content'):
            if isinstance(result.content, list):
                out = [getattr(item, 'text', str(item)) for item in result.content]
            else:
                out = getattr(result.content, 'text', str(result.content))
        else:
            out = str(result)
        
        log("tool", f"✅ {tool_name} result: {out}")
        return ToolCallResult(
            tool_name=tool_name,
            arguments=args,
            result=out,
            raw_response=result
        )
        
    except Exception as e:
        log("tool", f"❌ Failed to execute tool {response}: {e}")
        raise
    
    