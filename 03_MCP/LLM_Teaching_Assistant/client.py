# client.py
import os
import asyncio
import json
from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

async def run_teaching_assistant():
    # Configure the MCP server location
    server_params = StdioServerParameters(
        command="python", 
        args=["server.py"]
    )
    
    print("Initializing JEE/NEET Teaching Guide Project...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 1. Fetch available tools from our MCP Server
            mcp_tools = await session.list_tools()
            
            # Map MCP tool schema to Groq tool schema format
            groq_formatted_tools = []
            for tool in mcp_tools.tools:
                groq_formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            
            # 2. Simple Mock User Interaction
            user_message = "I am stuck on a Physics Kinematics numerical problem. Can you help me with the formulas?"
            print(f"\nStudent: {user_message}")
            
            # 3. Call Groq (Using Llama-3.1-70b-Versatile or 8b for fast free tier responses)
            messages = [{"role": "user", "content": user_message}]
            
            response = groq_client.chat.completions.create(
                model="llama3-8b-8192", # Free-tier stable model
                messages=messages,
                tools=groq_formatted_tools,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            
            # 4. Handle Tool Calls requested by the LLM
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"LLM executing MCP Tool [{tool_name}] with args: {tool_args}")
                    
                    # Execute tool on our local MCP server
                    tool_result = await session.call_tool(tool_name, arguments=tool_args)
                    print(f"Data retrieved from local repository.")
                    
                    # Send results back to Groq for final pedagogical synthesis
                    messages.append(response_message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": str(tool_result.content)
                    })
                    
                    # Final call to Groq to generate the tutor's response
                    final_response = groq_client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=messages
                    )
                    print(f"\nTutor AI:\n{final_response.choices[0].message.content}")
            else:
                print(f"\nTutor AI:\n{response_message.content}")

if __name__ == "__main__":
    asyncio.run(run_teaching_assistant())