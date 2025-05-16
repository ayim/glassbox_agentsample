# tools.py
from langgraph import Tool

class MockDocumentLoaderTool(Tool):
    def name(self): 
        return "DocumentLoaderTool"
    def call(self, docs):
        print(f"[MOCK] DocumentLoaderTool ingesting docs: {docs}")
        return {"documents": docs}

class MockHTSLookupTool(Tool):
    def name(self): 
        return "HTSLookupTool"
    def call(self, hs_codes):
        print(f"[MOCK] HTSLookupTool called with: {hs_codes}")
        return {
            "results": [
                {"code": c, "tariff_rate": 0.05, "cfr_violations": []}
                for c in hs_codes
            ]
        }

class MockESGScoringTool(Tool):
    def name(self): 
        return "ESGScoringAPI"
    def call(self, details):
        print(f"[MOCK] ESGScoringAPI received case details: {details}")
        return {"score": 88}

class MockVendorSQLTool(Tool):
    def name(self): 
        return "SQLTool: ApprovedVendorDB"
    def call(self, query):
        print(f"[MOCK] SQLTool queried with: {query}")
        return {"vendors": [{"id": 1, "name": "MockVendor"}]}

class MockTMSPushTool(Tool):
    def name(self): 
        return "TMSPushTool"
    def call(self, payload):
        print(f"[MOCK] TMSPushTool got payload: {payload}")
        return {"status": "mocked_success", "id": "TMS-999"}

class MockVectorRetriever(Tool):
    def __init__(self, namespace=None):
        self.namespace = namespace
    def name(self):
        return f"VectorRetriever:{self.namespace}"
    def call(self, query):
        print(f"[MOCK] VectorRetriever({self.namespace}) called with: {query}")
        if self.namespace == "StyleGuide":
            docs = ["Missing HS Code section", "Wrong naming convention"]
        elif self.namespace == "Regulations":
            docs = ["CFR §123.45 applies", "HS code valid"]
        else:
            docs = ["Generic doc"]
        return {"documents": docs}

class MockSlackTool(Tool):
    def name(self):
        return "SlackTool"
    def call(self, channel, message):
        print(f"[MOCK] SlackTool posting to {channel}: {message}")
        return {"ts": "1234567890.123456"}

class MockEmailTool(Tool):
    def name(self):
        return "EmailTool"
    def call(self, to, subject, body):
        print(f"[MOCK] EmailTool sending to {to}, subject={subject}")
        return {"message_id": "msg-001"}

class MockErrorTool(Tool):
    def name(self):
        return "ErrorTool"
    def call(self, *args, **kwargs):
        print("[MOCK] ErrorTool intentionally raising error")
        raise Exception("This is a simulated error from MockErrorTool")

class MockCalendarTool(Tool):
    def name(self):
        return "CalendarTool"
    def call(self, event_details):
        print(f"[MOCK] CalendarTool received event details: {event_details}")
        raise Exception("Calendar service is currently unavailable")