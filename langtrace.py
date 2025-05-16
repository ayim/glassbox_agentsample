from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from os import getenv
from dotenv import load_dotenv
from time import sleep
import json
import os
# Import mock inputs for testing
try:
    from tests.mock_inputs import MOCK_ROUTER_RESPONSE, MOCK_CASE_1
except ImportError:
    MOCK_ROUTER_RESPONSE = None
# Import mock input for case 1
try:
    from tests.mock_inputs import MOCK_CASE_1
except ImportError:
    MOCK_CASE_1 = "Sample mock case 1 input."

load_dotenv()

openai_client = wrap_openai(OpenAI())

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"

@traceable(run_type="retriever")
def retriever(query: str):
    results = ["Harrison worked at Kensho"]
    return results

@traceable(run_type="tool")
def search_latest_knowledge(query: str):
    sleep(0.2)
    hscode_results_1 = lookup_HSCode_details("0101.21")
    hscode_results_2 = lookup_HSCode_details("0202.30")
    hscode_results_3 = lookup_HSCode_details("0303.40")
    return ["Latest knowledge about: " + query] + hscode_results_1 + hscode_results_2 + hscode_results_3

@traceable
def VectorStoreRetriever(query: str):
    sleep(0.15)
    return ["Vector store result for: " + query]

@traceable(run_type="tool")
def websearch_latest_knowledge(query: str):
    return ["Tariff impact for: " + query]

@traceable(name="search_HSCode_details", run_type="retriever")
def lookup_HSCode_details(code: str):
    sleep(0.3)
    return [f"HSCode search result for: {code}"]

@traceable(name="pdf_extract_text", run_type="retriever")
def extract_text_from_pdf(file: str):
    sleep(0.4)
    return f"Parsed content of {file}"

@traceable(name="batch_process_client_docs")
def process_client_submissions(files):
    sleep(0.1)
    parsed_docs = []
    for idx, file in enumerate(files):
        content = extract_text_from_pdf(file)
        doc = {
            "PageContent": content,
            "Metadata": {
                "Loc": {
                    "Lines": {
                        "From": 10 * idx + 1,
                        "To": 10 * idx + 5
                    }
                }
            }
        }
        parsed_docs.append(doc)
    return parsed_docs

@traceable(name="ChatPromptTemplate", run_type="prompt")
def chat_prompt_template(system_message, question):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question},
    ]

@traceable(run_type="llm")
def call_llm(messages):
    return openai_client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
    )

@traceable(name="RunnableLambda")
def parse_tool_output(tool_output):
    # Simulate parsing tool output
    sleep(0.1)
    return f"Parsed: {tool_output}"

@traceable(name="RunnableMap")
def decide_and_call_tool(question, tool_list):
    prompt = f"Given the question: '{question}', and these tools: {tool_list}, which tool(s) should be used? Respond with a comma-separated list of tool names, or a JSON list if you prefer."
    messages = chat_prompt_template(prompt, question)
    llm_response = chat_openai(messages)
    # Try to parse as JSON list first
    if hasattr(llm_response, 'choices'):
        content = llm_response.choices[0].message.content.strip()
    elif isinstance(llm_response, dict) and 'choices' in llm_response:
        content = llm_response['choices'][0]['message']['content'].strip()
    else:
        content = str(llm_response).strip()
    try:
        # Try to parse as JSON list
        tool_decisions = json.loads(content)
        if isinstance(tool_decisions, list):
            return [t.strip() for t in tool_decisions]
    except Exception:
        # Fallback: split by comma or newline
        if ',' in content:
            return [t.strip() for t in content.split(',') if t.strip()]
        else:
            return [t.strip() for t in content.split('\n') if t.strip()]

@traceable(name="Chat", run_type="llm")
def chat_openai(messages):
    if USE_MOCK and MOCK_ROUTER_RESPONSE:
        class MockResponse:
            def __init__(self, content):
                self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': content})})()]
        return MockResponse(MOCK_ROUTER_RESPONSE)
    return openai_client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
    )

@traceable(name="AgentOutputParser", run_type="parser")
def agent_output_parser(llm_response):
    # Accepts the real LLM response object, extracts the content, and parses as JSON
    try:
        # Get the content string from the real OpenAI response
        if hasattr(llm_response, 'choices'):
            content = llm_response.choices[0].message.content
        elif isinstance(llm_response, dict) and 'choices' in llm_response:
            content = llm_response['choices'][0]['message']['content']
        else:
            content = str(llm_response)
        data = json.loads(content)
        return {
            "chain_of_thought": data.get("chain_of_thought", ""),
            "routing_decision": data.get("routing_decision", []),
            "confidences": data.get("confidences", {})
        }
    except Exception:
        return {
            "chain_of_thought": "Could not parse LLM output.",
            "routing_decision": [],
            "confidences": {}
        }

@traceable(name="RoutingDecisionParser")
def parse_routing_decision(parsed_llm):
    # Actually parse the LLM output for reasoning, routing, and confidences
    # Assume the LLM output is a JSON string with the required fields
    try:
        data = json.loads(parsed_llm)
        return {
            "chain_of_thought": data.get("chain_of_thought", ""),
            "routing_decision": data.get("routing_decision", []),
            "confidences": data.get("confidences", {})
        }
    except Exception as e:
        return {
            "chain_of_thought": "Could not parse LLM output.",
            "routing_decision": [],
            "confidences": {}
        }

@traceable(name="AgentHandoff")
def agent_handoff(agent_name, docs):
    agent = get_tool_by_name(agent_name)
    if agent:
        return agent["function"](docs)
    return f"No agent found for {agent_name}"

# Tool definitions (only actual tools)
TOOLS = [
    {
        "name": "search_latest_knowledge",
        "description": "Searches the latest knowledge base for relevant information.",
        "function": search_latest_knowledge,
    },
    {
        "name": "vector_store_retriever",
        "description": "Retrieves relevant information from the vector store based on the query.",
        "function": VectorStoreRetriever,
    },
    {
        "name": "batch_process_client_docs",
        "description": "Processes and extracts information from all client-uploaded PDF documents.",
        "function": process_client_submissions,
    },
    {
        "name": "lookup_HSCode_details",
        "description": "Looks up details and regulations for a given HSCode.",
        "function": lookup_HSCode_details,
    },
]

# Agent definitions (handoff targets, not tools)
AGENTS = [
    {
        "name": "declaration_review",
        "description": "Use when product classification is unclear, documentation is incomplete or ambiguous, or multiple HS/HTS codes apply.",
    },
    {
        "name": "regulatory_sustainability",
        "description": "Use when the item may violate import/export laws, require special licenses, or involve ESG, REACH, CBAM, or other regulatory compliance concerns.",
    },
    {
        "name": "sourcing_logistics",
        "description": "Use when everything is in order and the shipment is routine, well-documented, and ready for normal processing.",
    },
]

# Helper to get tool by name

def get_tool_by_name(name):
    for tool in TOOLS:
        if tool["name"] == name:
            return tool
    return None

@traceable(name="OrchestratorAgent")
def orchestrator_agent(question, tool_outputs=None):
    tool_descriptions = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in TOOLS])
    agent_descriptions = "\n".join([f"- {agent['name']}: {agent['description']}" for agent in AGENTS])
    allowed_agents = ', '.join([agent['name'] for agent in AGENTS])
    base_prompt = (
        "You are the router agent. Given the question and the input (from mock input), decide whether we should call one or multiple tools to learn more about the situation or if we are ready to make a decision on a handoff.\n"
        f"When you are ready to make a final decision, your routing_decision must be one of the following agent handoffs: {allowed_agents}, or 'done' if no further action is needed. Do not invent new agent names.\n"
        "For each possible agent, provide a confidence score (0-1) for how appropriate it is for this case.\n"
        "Return your answer as a JSON object with keys: chain_of_thought (string), routing_decision (list of agent names), confidences (dict of agent name to confidence float).\n"
        "Let's think step by step:\n"
        "- Review document types and content\n"
        "- Determine if more tool calls are needed\n"
        "- Determine if a handoff is needed and to which agent\n"
        f"\nAvailable tools:\n{tool_descriptions}\nAvailable agents for handoff:\n{agent_descriptions}"
    )
    plan = decide_and_call_tool(question, [tool['name'] for tool in TOOLS])
    # Actually call the selected tool(s) outside of decide_and_call_tool
    tool_output = None
    if plan == "search_latest_knowledge":
        tool_output = search_latest_knowledge(question)
    elif plan == "vector_store_retriever":
        tool_output = VectorStoreRetriever(question)
    elif plan == "batch_process_client_docs":
        tool_output = process_client_submissions(["file1.pdf", "file2.pdf", "file3.pdf"])
    elif plan == "lookup_HSCode_details":
        tool_output = lookup_HSCode_details("0101.21")
    # Add more tool calls as needed
    parsed_plan = parse_tool_output(plan)
    system_message = base_prompt + f"\n\nQuestion: {question}"
    messages = chat_prompt_template(system_message, question)
    llm_response = chat_openai(messages)
    parsed_llm = agent_output_parser(llm_response)
    # Only expose the decision, do not execute handoff
    return {
        "plan": plan,
        "parsed_plan": parsed_plan,
        "tool_output": tool_output,
        "llm_response": llm_response,
        "parsed_llm": parsed_llm,
        "chain_of_thought": parsed_llm["chain_of_thought"],
        "routing_decision": parsed_llm["routing_decision"],
        "confidences": parsed_llm["confidences"]
    }

@traceable(name="RunnableAgent")
def runnable_agent(question):
    context = {"question": question, "tool_outputs": {}}
    trajectory = []
    max_steps = 10
    steps = 0
    allowed_agents = [a["name"].lower() for a in AGENTS]
    while steps < max_steps:
        orchestrator_result = orchestrator_agent(context["question"], context["tool_outputs"])
        trajectory.append({
            "orchestrator_result": orchestrator_result,
            "tool_outputs": dict(context["tool_outputs"]),
        })
        routing_decision = orchestrator_result.get("routing_decision", [])
        # Flexible done criteria: all agent names (case-insensitive, stripped) or 'done'
        normalized = [a.lower().strip() for a in routing_decision]
        if all(agent in allowed_agents for agent in normalized) or normalized == ["done"]:
            break
        for tool_name in routing_decision:
            if tool_name in context["tool_outputs"]:
                continue
            if tool_name == "search_latest_knowledge":
                context["tool_outputs"][tool_name] = search_latest_knowledge(question)
            elif tool_name == "vector_store_retriever":
                context["tool_outputs"][tool_name] = VectorStoreRetriever(question)
            elif tool_name == "batch_process_client_docs":
                context["tool_outputs"][tool_name] = process_client_submissions(["file1.pdf", "file2.pdf", "file3.pdf"])
            elif tool_name == "lookup_HSCode_details":
                context["tool_outputs"][tool_name] = lookup_HSCode_details("0101.21")
        steps += 1
    return {
        "trajectory": trajectory,
        "final_decision": orchestrator_result
    }

if __name__ == "__main__":
    result = runnable_agent(MOCK_CASE_1)
    print(result)