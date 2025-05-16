from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from os import getenv
from dotenv import load_dotenv
from time import sleep
load_dotenv()

openai_client = wrap_openai(OpenAI())

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
def decide_and_call_tool(question):
    # Simulate agent deciding which tool to use
    sleep(0.1)
    # For this example, always use search_latest_knowledge
    tool_output = search_latest_knowledge(question)
    return tool_output

@traceable(name="Chat", run_type="llm")
def chat_openai(messages):
    sleep(0.2)
    return {"choices": [{"message": {"content": "Simulated LLM response"}}]}

@traceable(name="OpenAIFunctionsAgentOutputParser", run_type="parser")
def openai_functions_agent_output_parser(llm_response):
    sleep(0.05)
    return f"Parsed LLM output: {llm_response['choices'][0]['message']['content']}"

@traceable(name="OrchestratorAgent")
def orchestrator_agent(question, tool_outputs=None):
    # First or second reasoning loop depending on tool_outputs
    plan = decide_and_call_tool(question)
    parsed_plan = parse_tool_output(plan)
    # Prompt construction
    system_message = f"System: {question}"
    messages = chat_prompt_template(system_message, question)
    llm_response = chat_openai(messages)
    parsed_llm = openai_functions_agent_output_parser(llm_response)
    return {
        "plan": plan,
        "parsed_plan": parsed_plan,
        "llm_response": llm_response,
        "parsed_llm": parsed_llm
    }

@traceable(name="RunnableAgent")
def runnable_agent(question):
    # First orchestrator call (reasoning loop)
    orchestrator_1 = orchestrator_agent(question)
    # Tool calls (simulate with current tools)
    docs = retriever(question)
    vector_results = VectorStoreRetriever(question)
    parsed_documents = process_client_submissions(["file1.pdf", "file2.pdf", "file3.pdf"])
    # Second orchestrator call (reasoning loop with tool outputs)
    tool_outputs = {
        "docs": docs,
        "vector_results": vector_results,
        "parsed_documents": parsed_documents
    }
    orchestrator_2 = orchestrator_agent(question + " (with tool outputs)", tool_outputs=tool_outputs)
    return orchestrator_2

if __name__ == "__main__":
    runnable_agent("Where did harrison even work?")