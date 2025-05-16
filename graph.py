# Writing graph.py file with the multi-agent logic and routing into the project directory

from langgraph import Graph, Node
from tools import (
    MockDocumentLoaderTool,
    MockHTSLookupTool,
    MockESGScoringTool,
    MockVendorSQLTool,
    MockTMSPushTool,
    MockVectorRetriever,
    MockSlackTool,
    MockEmailTool,
    MockCalendarTool,
)

def build_graph(env="dev"):
    # Instantiate the graph
    g = Graph(name="SupplyChainWorkflow")
    
    # Register tools
    if env == "dev":
        g.register_tool(MockDocumentLoaderTool())
        g.register_tool(MockHTSLookupTool())
        g.register_tool(MockESGScoringTool())
        g.register_tool(MockVendorSQLTool())
        g.register_tool(MockTMSPushTool())
        g.register_tool(MockVectorRetriever(namespace="StyleGuide"))
        g.register_tool(MockVectorRetriever(namespace="Regulations"))
        g.register_tool(MockVectorRetriever(namespace="Default"))
        g.register_tool(MockSlackTool())
        g.register_tool(MockEmailTool())
        g.register_tool(MockCalendarTool())
    else:
        # TODO: register real production tools here
        pass

    # Define nodes
    loader = Node(
        id="DocumentLoader",
        type="loader",
        tool="DocumentLoaderTool",
        description="Ingest case documents"
    )

    router_prompt = """
You are the router agent. Given the case documents, classify and decide which agents to run. Provide both your internal reasoning (chain_of_thought) and your routing decision.
Let's think step by step:
- Review document types and content
- Determine if general doc review is always needed
- Determine if regulatory/sustainability review is needed
- Determine if sourcing/logistics handoff is needed
Respond with JSON:
{
  "chain_of_thought": ["..."],
  "decision": {"run_agentA": true, "run_agentB": <boolean>, "run_agentC": <boolean>},
  "user_summary": "I recommend running AgentA always, AgentB only if regs needed, AgentC only if sourcing needed."
}
"""
    router = Node(
        id="GraphRouter",
        type="router",
        prompt=router_prompt
    )

    agentA_prompt = """
You are a document auditor. Let's think step by step:
1) Compare the case file against our style guide: required sections, naming conventions, attachments.
2) Scan for incomplete or ambiguous declaration fields (e.g., missing unit, fuzzy quantity).
After your internal reasoning, provide:
- chain_of_thought: an array of your reasoning steps
- decision: { "ambiguities_found": [...], "escalate": true/false }
- user_summary: a concise, user-friendly explanation of findings and next steps
"""
    agentA = Node(
        id="AgentA",
        type="agent",
        prompt=agentA_prompt,
        tools=["VectorRetriever:StyleGuide", "SlackTool", "EmailTool"]
    )

    agentB_prompt = """
You are a compliance & sustainability analyst. Let's think step by step:
1) For each HS code, fetch tariff rates & check CFR compliance.
2) Estimate the carbon footprint using the ESG API.
After your reasoning, provide:
- chain_of_thought: your step-by-step analysis
- decision: { "reg_flags": [...], "sustainability_score": <number>, "escalate": true/false }
- user_summary: a clear, non-technical summary of any risks and recommendations
"""
    agentB = Node(
        id="AgentB",
        type="agent",
        prompt=agentB_prompt,
        tools=["HTSLookupTool", "VectorRetriever:Regulations", "ESGScoringAPI"]
    )

    agentC_prompt = """
You are a sourcing/logistics coordinator. Let's think step by step:
1) Query the approved-vendor DB and evaluate best-fit options.
2) Generate an RFQ payload/email with specs, qty, and target date.
3) Schedule a calendar follow-up in 48 hours.
4) Push order details to the TMS.
After your reasoning, provide:
- chain_of_thought: reasoning steps
- decision: { "vendor_id": <id>, "rfq_sent": true, "follow_up_scheduled": true }
- tms_payload: {...}
- user_summary: a brief update on next logistic steps and owner
"""
    agentC = Node(
        id="AgentC",
        type="agent",
        prompt=agentC_prompt,
        tools=["SQLTool:ApprovedVendorDB", "CalendarTool", "TMSPushTool"]
    )

    error_handler = Node(
        id="ErrorHandler",
        type="handler",
        tools=["SlackTool", "EmailTool"],
        description="Escalates failures to humans"
    )

    # Add nodes to graph
    for node in [loader, router, agentA, agentB, agentC, error_handler]:
        g.add_node(node)

    # Connect edges
    g.add_edge(loader, router)
    g.add_edge_conditional(router, agentA, condition="True")  # always run A
    g.add_edge_conditional(router, agentB, condition="decision.get('run_agentB', False)")
    g.add_edge_conditional(router, agentC, condition="decision.get('run_agentC', False)")

    # Error handling
    g.on_error(error_handler)

    return g

# Write to file
with open('/mnt/data/graph.py', 'w') as f:
    f.write(graph_py)

# Provide link for user
