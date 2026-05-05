import os
import base64
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates


load_dotenv()

app = FastAPI(title="Jira Ticket Quality Compliance Tool")

templates = Jinja2Templates(directory="templates")

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "").rstrip("/")
JIRA_EMAIL = os.getenv("JIRA_EMAIL", "")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")
DEFAULT_PROJECT = os.getenv("JIRA_PROJECT_KEY", "ARC").strip().upper()


def jira_auth_headers() -> Dict[str, str]:
    token = f"{JIRA_EMAIL}:{JIRA_API_TOKEN}"
    encoded = base64.b64encode(token.encode()).decode()
    return {
        "Authorization": f"Basic {encoded}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def normalize_project_key(project_key: Optional[str]) -> str:
    key = (project_key or DEFAULT_PROJECT).strip().upper()
    return key or DEFAULT_PROJECT


def extract_plain_text_from_adf(adf: Any) -> str:
    """
    Jira Cloud descriptions are often stored in Atlassian Document Format (ADF).
    This function flattens that structure into plain text.
    """
    if not adf:
        return ""

    texts: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type == "text":
                texts.append(node.get("text", ""))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(adf)
    return " ".join(part.strip() for part in texts if str(part).strip()).strip()


def build_adf_paragraph(text: str) -> Dict[str, Any]:
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": text or ""}],
            }
        ],
    }


def safe_json_response(response: requests.Response) -> Dict[str, Any]:
    try:
        body = response.json()
    except Exception:
        body = {"raw_text": response.text}
    return {
        "status_code": response.status_code,
        "jira_response": body,
    }


def score_ticket_quality(issue_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple deterministic scoring model.
    Works for:
    - manual payloads
    - live Jira issues normalized into this structure
    """
    summary = (issue_data.get("summary") or "").strip()
    description = (issue_data.get("description") or "").strip()
    issue_type = (issue_data.get("issue_type") or "").strip()
    labels = issue_data.get("labels") or []
    components = issue_data.get("components") or []
    priority = (issue_data.get("priority") or "").strip()
    dependencies = issue_data.get("dependencies") or []
    acceptance_criteria = issue_data.get("acceptance_criteria") or []

    score = 100
    issues = []
    strengths = []

    # Summary
    if len(summary) < 8:
        score -= 20
        issues.append("Summary is too short.")
    elif len(summary) < 20:
        score -= 8
        issues.append("Summary is present but not very specific.")
    else:
        strengths.append("Summary is reasonably specific.")

    # Description
    if len(description) == 0:
        score -= 30
        issues.append("Description is missing.")
    elif len(description) < 40:
        score -= 20
        issues.append("Description is too short.")
    elif len(description) < 120:
        score -= 8
        issues.append("Description exists but could use more implementation detail.")
    else:
        strengths.append("Description provides usable detail.")

    # Issue type
    if not issue_type:
        score -= 5
        issues.append("Issue type is missing.")
    else:
        strengths.append(f"Issue type is set to {issue_type}.")

    # Acceptance criteria
    ac_text = " ".join(acceptance_criteria).strip() if isinstance(acceptance_criteria, list) else str(acceptance_criteria).strip()
    if not ac_text:
        if "given" in description.lower() and "when" in description.lower() and "then" in description.lower():
            strengths.append("Acceptance criteria appear to be embedded in the description.")
        else:
            score -= 20
            issues.append("Acceptance criteria are missing.")
    else:
        lower_ac = ac_text.lower()
        if "given" in lower_ac and "when" in lower_ac and "then" in lower_ac:
            strengths.append("Acceptance criteria are present.")
        else:
            score -= 8
            issues.append("Acceptance criteria exist but are not well structured.")

    # Dependencies
    dep_text = " ".join(dependencies).strip() if isinstance(dependencies, list) else str(dependencies).strip()
    if dep_text:
        strengths.append("Dependencies are documented.")
    else:
        if "dependency" in description.lower() or "depends on" in description.lower():
            strengths.append("Dependencies appear to be mentioned in the description.")
        else:
            score -= 10
            issues.append("Dependencies are not documented.")

    # Labels / components
    if labels:
        strengths.append("Labels are present.")
    else:
        score -= 5
        issues.append("Labels are missing.")

    if components:
        strengths.append("Components are present.")
    else:
        score -= 5
        issues.append("Components are missing.")

    # Priority
    if priority:
        strengths.append(f"Priority is set to {priority}.")
    else:
        score -= 3
        issues.append("Priority is missing.")

    score = max(0, min(score, 100))

    if score >= 80:
        status = "PASS"
        compliance = "Strong"
    elif score >= 60:
        status = "PARTIAL"
        compliance = "Moderate"
    else:
        status = "FAIL"
        compliance = "Weak"

    suggested_rewrite = f"""Business need: {summary or 'Describe the business objective clearly'}.

Detailed context:
Provide the purpose, target systems, business driver, and expected delivery outcome.

Dependencies:
- List upstream or downstream dependencies
- List approvals or external teams required

Acceptance Criteria:
- Given a valid request, when the work is executed, then the expected technical outcome is achieved.
- Given an invalid condition, when the work is attempted, then the issue is rejected or handled correctly.
- Given completion, when validation is performed, then evidence confirms successful delivery.
"""

    return {
        "score": score,
        "status": status,
        "compliance": compliance,
        "issues": issues,
        "strengths": strengths,
        "suggested_rewrite": suggested_rewrite.strip(),
    }


def normalize_live_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
    fields = issue.get("fields", {}) or {}

    description_text = extract_plain_text_from_adf(fields.get("description"))

    return {
        "issue_key": issue.get("key"),
        "summary": fields.get("summary") or "",
        "description": description_text,
        "issue_type": (fields.get("issuetype") or {}).get("name", ""),
        "labels": fields.get("labels") or [],
        "components": [c.get("name", "") for c in (fields.get("components") or []) if isinstance(c, dict)],
        "priority": (fields.get("priority") or {}).get("name", ""),
        "status": (fields.get("status") or {}).get("name", ""),
        "assignee": (fields.get("assignee") or {}).get("displayName", ""),
        "reporter": (fields.get("reporter") or {}).get("displayName", ""),
        "created": fields.get("created", ""),
        "updated": fields.get("updated", ""),
        "dependencies": [],
        "acceptance_criteria": [],
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_project": DEFAULT_PROJECT,
        },
    )


@app.get("/api/health")
def api_health():
    return {
        "ok": True,
        "jira_base_url": JIRA_BASE_URL,
        "default_project": DEFAULT_PROJECT,
    }


@app.get("/api/jira/me")
def jira_me():
    url = f"{JIRA_BASE_URL}/rest/api/3/myself"
    response = requests.get(url, headers=jira_auth_headers(), timeout=30)
    return safe_json_response(response)


@app.post("/api/jira/search")
def jira_search(body: Dict[str, Any]):
    project_key = normalize_project_key(body.get("projectKey"))
    jql = body.get("jql") or f"project = {project_key} ORDER BY created DESC"
    max_results = int(body.get("maxResults", 10))
    next_page_token = body.get("nextPageToken")
    fields = body.get(
        "fields",
        [
            "summary",
            "description",
            "status",
            "assignee",
            "issuetype",
            "labels",
            "priority",
            "components",
            "created",
            "updated",
            "reporter",
        ],
    )

    url = f"{JIRA_BASE_URL}/rest/api/3/search/jql"
    params = {
        "jql": jql,
        "maxResults": max_results,
        "fields": ",".join(fields),
        "fieldsByKeys": "false",
    }

    if next_page_token:
        params["nextPageToken"] = next_page_token

    response = requests.get(
        url,
        headers=jira_auth_headers(),
        params=params,
        timeout=60,
    )

    result = safe_json_response(response)
    result["projectKey"] = project_key
    result["jql"] = jql
    return result


@app.get("/api/jira/issue/{issue_key}")
def jira_get_issue(issue_key: str):
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}"
    params = {
        "fields": "summary,description,status,assignee,issuetype,labels,priority,components,created,updated,reporter"
    }
    response = requests.get(
        url,
        headers=jira_auth_headers(),
        params=params,
        timeout=30,
    )
    return safe_json_response(response)


@app.post("/api/tickets/check")
def check_ticket(payload: Dict[str, Any]):
    data = {
        "summary": payload.get("summary", ""),
        "description": payload.get("description", ""),
        "issue_type": payload.get("issue_type") or payload.get("issueType") or "",
        "labels": payload.get("labels", []),
        "components": payload.get("components", []),
        "priority": payload.get("priority", ""),
        "dependencies": payload.get("dependencies", []),
        "acceptance_criteria": payload.get("acceptance_criteria", []),
    }
    return score_ticket_quality(data)


@app.post("/api/tickets/check-live")
def check_live_ticket(payload: Dict[str, Any]):
    issue_key = (payload.get("issueKey") or "").strip().upper()
    if not issue_key:
        return JSONResponse(
            status_code=400,
            content={"error": "issueKey is required, for example ARC-827 or NMGOS-3846"},
        )

    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}"
    params = {
        "fields": "summary,description,status,assignee,issuetype,labels,priority,components,created,updated,reporter"
    }

    response = requests.get(
        url,
        headers=jira_auth_headers(),
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        return JSONResponse(
            status_code=response.status_code,
            content=safe_json_response(response),
        )

    issue = response.json()
    normalized = normalize_live_issue(issue)
    assessment = score_ticket_quality(normalized)

    return {
        "issueKey": issue_key,
        "normalized_issue": normalized,
        "assessment": assessment,
    }


@app.post("/api/tickets/rewrite")
def rewrite_ticket(payload: Dict[str, Any]):
    summary = (payload.get("summary") or "").strip()
    description = (payload.get("description") or "").strip()

    improved = f"""Business need:
{summary or "State the business objective clearly."}

Detailed context:
{description or "Add background, target systems, scope, and expected outcome."}

Dependencies:
- Identify dependent systems, teams, approvals, or inputs.
- State any security, architecture, or firewall dependencies.

Acceptance Criteria:
- Given a valid request, when the work is executed, then the expected technical outcome is achieved.
- Given an invalid request or missing prerequisite, when the work is attempted, then a controlled error or dependency outcome is returned.
- Given implementation completion, when validation is performed, then success evidence is available and documented.
"""

    return {
        "summary": summary,
        "original_description": description,
        "rewritten_description": improved.strip(),
    }


@app.post("/api/jira/create")
def jira_create(data: Dict[str, Any]):
    project_key = normalize_project_key(data.get("projectKey"))
    issue_type = data.get("issueType", "Story")
    summary = data.get("summary", "").strip()
    description = data.get("description", "").strip()

    if not summary:
        return JSONResponse(
            status_code=400,
            content={"error": "summary is required"},
        )

    url = f"{JIRA_BASE_URL}/rest/api/3/issue"

    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": build_adf_paragraph(description),
            "issuetype": {"name": issue_type},
        }
    }

    response = requests.post(
        url,
        headers=jira_auth_headers(),
        json=payload,
        timeout=30,
    )

    result = safe_json_response(response)
    result["projectKey"] = project_key
    return result