from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


APP_TITLE = "Rovo-Integrated Compliance Platform"
APP_VERSION = "1.0.0"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# -----------------------------
# Models
# -----------------------------
class JiraQueryRequest(BaseModel):
    jql: str = Field(default="project = OSS ORDER BY updated DESC")
    max_results: int = Field(default=25, ge=1, le=100)


class ComplianceCheckRequest(BaseModel):
    issues: List[Dict[str, Any]]


class TicketDraftRequest(BaseModel):
    summary: str
    business_context: str
    integration_type: Optional[str] = "API"
    api_standard: Optional[str] = "TMF641"
    data_source: Optional[str] = "Oracle UIM"
    dependencies: Optional[List[str]] = None
    issue_type: Optional[str] = "Story"


class RovoWebhookRequest(BaseModel):
    action: str = Field(description="check_compliance | draft_ticket | summarize")
    payload: Dict[str, Any]


@dataclass
class RuleResult:
    rule_id: str
    description: str
    status: str
    weight: int
    message: str


# -----------------------------
# Jira client
# -----------------------------
class JiraClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("JIRA_BASE_URL", "").rstrip("/")
        self.email = os.getenv("JIRA_EMAIL", "")
        self.api_token = os.getenv("JIRA_API_TOKEN", "")
        self.project_key = os.getenv("JIRA_PROJECT_KEY", "OSS")

    def configured(self) -> bool:
        return bool(self.base_url and self.email and self.api_token)

    def _headers(self) -> Dict[str, str]:
        return {"Accept": "application/json", "Content-Type": "application/json"}

    def _auth(self) -> tuple[str, str]:
        return (self.email, self.api_token)

    def search_issues(self, jql: str, max_results: int = 25) -> Dict[str, Any]:
        if not self.configured():
            raise HTTPException(status_code=400, detail="Jira credentials are not configured.")
        url = f"{self.base_url}/rest/api/3/search/jql"
        payload = {
            "jql": jql,
            "maxResults": max_results,
            "fields": [
                "summary",
                "description",
                "issuetype",
                "labels",
                "status",
                "assignee",
                "priority",
                "customfield_10001",  # Example placeholder field for Epic Link
            ],
        }
        response = requests.post(url, headers=self._headers(), auth=self._auth(), json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def create_issue(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        if not self.configured():
            raise HTTPException(status_code=400, detail="Jira credentials are not configured.")
        url = f"{self.base_url}/rest/api/3/issue"
        response = requests.post(url, headers=self._headers(), auth=self._auth(), json={"fields": fields}, timeout=30)
        response.raise_for_status()
        return response.json()


jira_client = JiraClient()


# -----------------------------
# Compliance logic
# -----------------------------
REQUIRED_FIELDS = [
    ("summary", "Summary must be populated", 15),
    ("description", "Description must explain the business and technical need", 20),
    ("integration_type", "Integration type must be declared", 15),
    ("api_standard", "API standard must be declared", 10),
    ("data_source", "Source system must be declared", 15),
    ("acceptance_criteria", "Acceptance criteria must be present", 15),
    ("dependencies", "Dependencies or blockers must be declared", 10),
]


def extract_text_description(description: Any) -> str:
    if description is None:
        return ""
    if isinstance(description, str):
        return description
    try:
        return json.dumps(description)
    except Exception:
        return str(description)


def derive_structured_fields(issue: Dict[str, Any]) -> Dict[str, Any]:
    fields = issue.get("fields", issue)
    summary = fields.get("summary", "")
    description = extract_text_description(fields.get("description"))
    labels = fields.get("labels", []) or []

    integration_type = fields.get("integration_type")
    api_standard = fields.get("api_standard")
    data_source = fields.get("data_source")
    dependencies = fields.get("dependencies")
    acceptance_criteria = fields.get("acceptance_criteria")

    # Lightweight inference from labels and description
    text = f"{summary}\n{description}\n{' '.join(labels)}".lower()
    if not integration_type:
        if "api" in text or "rest" in text or "restconf" in text:
            integration_type = "API"
        elif "file" in text or "csv" in text or "batch" in text:
            integration_type = "Batch/File"
    if not api_standard:
        if "tmf641" in text:
            api_standard = "TMF641"
        elif "tmf638" in text:
            api_standard = "TMF638"
        elif "restconf" in text:
            api_standard = "RESTCONF"
    if not data_source:
        for source in ["oracle uim", "smallworld", "nokia nsp", "huawei u2020", "sap"]:
            if source in text:
                data_source = source.title()
                break
    if not acceptance_criteria and "acceptance criteria" in text:
        acceptance_criteria = "Present in body"
    if not dependencies and ("dependency" in text or "blocker" in text):
        dependencies = "Mentioned in body"

    return {
        "summary": summary,
        "description": description,
        "integration_type": integration_type,
        "api_standard": api_standard,
        "data_source": data_source,
        "acceptance_criteria": acceptance_criteria,
        "dependencies": dependencies,
    }


def evaluate_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
    structured = derive_structured_fields(issue)
    results: List[RuleResult] = []
    total_weight = sum(weight for _, _, weight in REQUIRED_FIELDS)
    earned = 0

    for field_name, description, weight in REQUIRED_FIELDS:
        value = structured.get(field_name)
        passed = bool(value)
        if passed:
            earned += weight
        results.append(
            RuleResult(
                rule_id=field_name,
                description=description,
                status="PASS" if passed else "FAIL",
                weight=weight,
                message=f"{field_name} {'present' if passed else 'missing'}",
            )
        )

    score = round((earned / total_weight) * 100, 2) if total_weight else 0.0
    risk = "Low" if score >= 85 else "Medium" if score >= 60 else "High"

    return {
        "issue_key": issue.get("key", issue.get("issue_key", "LOCAL-1")),
        "summary": structured.get("summary", ""),
        "score": score,
        "risk": risk,
        "structured": structured,
        "rule_results": [asdict(r) for r in results],
        "recommendation": build_recommendation(structured),
    }


def build_recommendation(structured: Dict[str, Any]) -> str:
    gaps = []
    if not structured.get("integration_type"):
        gaps.append("declare the integration type")
    if not structured.get("api_standard"):
        gaps.append("reference the governing API standard")
    if not structured.get("data_source"):
        gaps.append("state the source or system of record")
    if not structured.get("acceptance_criteria"):
        gaps.append("add acceptance criteria")
    if not structured.get("dependencies"):
        gaps.append("capture upstream dependencies")
    if not gaps:
        return "Issue is largely ready. Confirm estimations, ownership, and test evidence before sprint commitment."
    return "Before the issue is treated as ready, " + ", ".join(gaps[:-1] + (["and " + gaps[-1]] if gaps else [])) + "."


# -----------------------------
# Drafting logic
# -----------------------------
def generate_ticket_draft(req: TicketDraftRequest) -> Dict[str, Any]:
    dependencies = req.dependencies or ["Architecture sign-off", "Upstream source-system availability"]
    description = f"""h2. Business Context\n{req.business_context}\n\nh2. Scope\nImplement {req.integration_type} capability aligned to {req.api_standard} with data originating from {req.data_source}.\n\nh2. Dependencies\n""" + "\n".join(f"* {d}" for d in dependencies) + "\n\nh2. Acceptance Criteria\n* Given valid input, when the transaction is submitted, then the platform validates mandatory fields and records audit data.\n* Given a downstream dependency failure, when processing occurs, then the platform returns a controlled error and logs the failure.\n* Given successful processing, when the workflow completes, then Jira fields and compliance evidence are updated.\n\nh2. Compliance Fields\n* Integration Type: {req.integration_type}\n* API Standard: {req.api_standard}\n* Data Source: {req.data_source}\n"""

    return {
        "issue_type": req.issue_type,
        "summary": req.summary,
        "description": description,
        "integration_type": req.integration_type,
        "api_standard": req.api_standard,
        "data_source": req.data_source,
        "dependencies": dependencies,
        "acceptance_criteria": "Embedded in description",
        "labels": ["rovo-generated", "compliance-ready"],
    }


# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    sample_issues = [
        {
            "key": "OSS-101",
            "fields": {
                "summary": "TMF641 integration for service order orchestration",
                "description": "Build REST API for service orders using TMF641 against Oracle UIM. Include acceptance criteria and dependency on HCO availability.",
                "labels": ["api", "tmf641", "oracle uim"],
            },
        },
        {
            "key": "OSS-102",
            "fields": {
                "summary": "Fix inventory sync",
                "description": "Need a fix soon.",
                "labels": [],
            },
        },
    ]
    report = [evaluate_issue(issue) for issue in sample_issues]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": APP_TITLE,
            "version": APP_VERSION,
            "report": report,
        },
    )


@app.post("/api/jira/search")
def api_jira_search(req: JiraQueryRequest):
    data = jira_client.search_issues(req.jql, req.max_results)
    return JSONResponse(data)


@app.post("/api/compliance/check")
def api_compliance_check(req: ComplianceCheckRequest):
    results = [evaluate_issue(issue) for issue in req.issues]
    avg = round(sum(item["score"] for item in results) / len(results), 2) if results else 0.0
    return {"count": len(results), "average_score": avg, "results": results}


@app.post("/api/tickets/draft")
def api_ticket_draft(req: TicketDraftRequest):
    return generate_ticket_draft(req)


@app.post("/api/jira/create")
def api_jira_create(req: TicketDraftRequest):
    draft = generate_ticket_draft(req)
    fields = {
        "project": {"key": jira_client.project_key},
        "summary": draft["summary"],
        "description": draft["description"],
        "issuetype": {"name": draft["issue_type"]},
        "labels": draft["labels"],
    }
    # Add custom fields when your Jira admin confirms the actual field IDs.
    created = jira_client.create_issue(fields)
    return {"draft": draft, "jira_response": created}


@app.post("/api/rovo/webhook")
def api_rovo_webhook(req: RovoWebhookRequest):
    action = req.action.lower().strip()
    payload = req.payload

    if action == "check_compliance":
        issues = payload.get("issues", [])
        results = [evaluate_issue(issue) for issue in issues]
        return {
            "message": "Compliance check completed.",
            "results": results,
            "summary": summarize_results(results),
        }

    if action == "draft_ticket":
        draft_req = TicketDraftRequest(**payload)
        draft = generate_ticket_draft(draft_req)
        return {
            "message": "Ticket draft generated for Rovo agent response.",
            "draft": draft,
        }

    if action == "summarize":
        results = payload.get("results", [])
        return {"summary": summarize_results(results)}

    raise HTTPException(status_code=400, detail=f"Unsupported action: {req.action}")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "app": APP_TITLE, "version": APP_VERSION}


def summarize_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No issues were supplied for compliance analysis."
    high_risk = [r for r in results if r.get("risk") == "High"]
    avg = round(sum(r.get("score", 0) for r in results) / len(results), 2)
    return (
        f"Reviewed {len(results)} issues. Average readiness score is {avg}%. "
        f"High-risk items: {len(high_risk)}. "
        "Focus first on missing acceptance criteria, undocumented dependencies, and absent integration metadata."
    )
