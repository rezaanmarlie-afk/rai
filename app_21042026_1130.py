import os
import re
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
    if not adf:
        return ""

    texts: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "text":
                texts.append(node.get("text", ""))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(adf)
    flattened = " ".join(part.strip() for part in texts if str(part).strip())
    flattened = re.sub(r"\s+", " ", flattened).strip()
    return flattened


def build_adf_document(text: str) -> Dict[str, Any]:
    text = text or ""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [""]

    content = []
    for para in paragraphs:
        content.append(
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": para}],
            }
        )

    return {
        "type": "doc",
        "version": 1,
        "content": content,
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


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def find_sentences_with_keywords(text: str, keywords: List[str], max_hits: int = 5) -> List[str]:
    hits: List[str] = []
    sentences = split_sentences(text)

    for sentence in sentences:
        lower = sentence.lower()
        if any(keyword.lower() in lower for keyword in keywords):
            hits.append(sentence)
        if len(hits) >= max_hits:
            break

    return hits


def find_section_snippet(text: str, headings: List[str], window: int = 1200) -> str:
    lower = text.lower()
    for heading in headings:
        idx = lower.find(heading.lower())
        if idx != -1:
            return text[idx: idx + window].strip()
    return ""


def detect_acceptance_evidence(description: str, explicit_ac: Any = None) -> Dict[str, Any]:
    evidence = []
    methods = []

    explicit_text = ""
    if isinstance(explicit_ac, list):
        explicit_text = " ".join(str(x) for x in explicit_ac if x)
    elif explicit_ac:
        explicit_text = str(explicit_ac)

    if explicit_text.strip():
        evidence.append(explicit_text.strip())
        methods.append("explicit_acceptance_criteria_field")

    lower = description.lower()

    if all(token in lower for token in ["given", "when", "then"]):
        evidence.extend(find_sentences_with_keywords(description, ["given", "when", "then"], max_hits=6))
        methods.append("given_when_then")

    for heading_group, method_name in [
        (["definition of ready", "dor"], "definition_of_ready"),
        (["definition of done", "dod"], "definition_of_done"),
        (["success criteria"], "success_criteria"),
        (["acceptance criteria"], "acceptance_criteria_heading"),
        (["ready when"], "ready_when"),
        (["done when"], "done_when"),
    ]:
        snippet = find_section_snippet(description, heading_group)
        if snippet:
            evidence.append(snippet)
            methods.append(method_name)

    return {
        "found": bool(methods),
        "methods": list(dict.fromkeys(methods)),
        "evidence": list(dict.fromkeys([e.strip() for e in evidence if e.strip()]))[:8],
    }


def detect_dependencies_evidence(description: str, explicit_dependencies: Any = None) -> Dict[str, Any]:
    evidence = []
    methods = []

    explicit_text = ""
    if isinstance(explicit_dependencies, list):
        explicit_text = " ".join(str(x) for x in explicit_dependencies if x)
    elif explicit_dependencies:
        explicit_text = str(explicit_dependencies)

    if explicit_text.strip():
        evidence.append(explicit_text.strip())
        methods.append("explicit_dependencies_field")

    dependency_keywords = [
        "dependency", "dependencies", "depends on", "required environments",
        "access and credentials", "vendor dependencies", "upstream", "downstream",
        "approval", "approvals", "security", "firewall", "access"
    ]

    dep_hits = find_sentences_with_keywords(description, dependency_keywords, max_hits=8)
    if dep_hits:
        evidence.extend(dep_hits)
        methods.append("description_dependency_mentions")

    snippet = find_section_snippet(description, ["inputs & dependencies", "dependencies"])
    if snippet:
        evidence.append(snippet)
        methods.append("dependency_section")

    return {
        "found": bool(methods),
        "methods": list(dict.fromkeys(methods)),
        "evidence": list(dict.fromkeys([e.strip() for e in evidence if e.strip()]))[:8],
    }


def detect_business_context(description: str) -> Dict[str, Any]:
    keywords = [
        "business problem", "overview", "initiative summary", "strategic value",
        "operational impact", "business need", "outcome", "results in"
    ]
    hits = find_sentences_with_keywords(description, keywords, max_hits=10)
    section = find_section_snippet(description, ["initiative summary", "overview", "business problem", "strategic value"])
    if section:
        hits.insert(0, section)
    unique = list(dict.fromkeys([h.strip() for h in hits if h.strip()]))
    return {"found": bool(unique), "evidence": unique[:8]}


def detect_scope(description: str) -> Dict[str, Any]:
    keywords = [
        "scope", "in scope", "out of scope", "use case", "use cases",
        "capabilities", "feature", "implements", "demonstrates"
    ]
    hits = find_sentences_with_keywords(description, keywords, max_hits=10)
    section = find_section_snippet(description, ["poc scope and capabilities", "scope", "scope & use cases"])
    if section:
        hits.insert(0, section)
    unique = list(dict.fromkeys([h.strip() for h in hits if h.strip()]))
    return {"found": bool(unique), "evidence": unique[:8]}


def detect_risk(description: str) -> Dict[str, Any]:
    keywords = [
        "risk", "risks", "mitigation", "rollback", "constraints",
        "operational risk", "security", "safe", "guardrails"
    ]
    hits = find_sentences_with_keywords(description, keywords, max_hits=10)
    section = find_section_snippet(description, ["non-functional & risk", "quality & safety", "risks"])
    if section:
        hits.insert(0, section)
    unique = list(dict.fromkeys([h.strip() for h in hits if h.strip()]))
    return {"found": bool(unique), "evidence": unique[:8]}


def detect_success_criteria(description: str) -> Dict[str, Any]:
    keywords = [
        "success criteria", "successful if", "measured", "improved",
        "reduction", "mttr", "effort", "thresholds", "go/no-go"
    ]
    hits = find_sentences_with_keywords(description, keywords, max_hits=10)
    section = find_section_snippet(description, ["success criteria", "success criteria & measurement", "successful if"])
    if section:
        hits.insert(0, section)
    unique = list(dict.fromkeys([h.strip() for h in hits if h.strip()]))
    return {"found": bool(unique), "evidence": unique[:8]}


def detect_dor_dod(description: str) -> Dict[str, Any]:
    dor = find_section_snippet(description, ["definition of ready", "dor"])
    dod = find_section_snippet(description, ["definition of done", "dod"])
    return {
        "dor_found": bool(dor),
        "dod_found": bool(dod),
        "dor_evidence": [dor] if dor else [],
        "dod_evidence": [dod] if dod else [],
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


def evaluate_ticket(issue_data: Dict[str, Any]) -> Dict[str, Any]:
    summary = (issue_data.get("summary") or "").strip()
    description = (issue_data.get("description") or "").strip()
    issue_type = (issue_data.get("issue_type") or issue_data.get("issueType") or "").strip()
    labels = issue_data.get("labels") or []
    components = issue_data.get("components") or []
    priority = (issue_data.get("priority") or "").strip()
    dependencies = issue_data.get("dependencies") or []
    acceptance_criteria = issue_data.get("acceptance_criteria") or []

    evidence = {
        "business_context": detect_business_context(description),
        "scope": detect_scope(description),
        "dependencies": detect_dependencies_evidence(description, dependencies),
        "acceptance": detect_acceptance_evidence(description, acceptance_criteria),
        "success_criteria": detect_success_criteria(description),
        "risk": detect_risk(description),
        "dor_dod": detect_dor_dod(description),
    }

    score = 100
    issues: List[str] = []
    strengths: List[str] = []

    if len(summary) < 8:
        score -= 20
        issues.append("Summary is too short.")
    elif len(summary) < 18:
        score -= 8
        issues.append("Summary is present but not very specific.")
    else:
        strengths.append("Summary is reasonably specific.")

    if len(description) == 0:
        score -= 35
        issues.append("Description is missing.")
    elif len(description) < 40:
        score -= 20
        issues.append("Description is too short.")
    elif len(description) < 120:
        score -= 8
        issues.append("Description exists but could use more delivery detail.")
    else:
        strengths.append("Description provides usable detail.")

    if not issue_type:
        score -= 5
        issues.append("Issue type is missing.")
    else:
        strengths.append(f"Issue type is set to {issue_type}.")

    if evidence["business_context"]["found"]:
        strengths.append("Business context is present.")
    else:
        score -= 8
        issues.append("Business context is not clearly stated.")

    if evidence["scope"]["found"]:
        strengths.append("Scope or use case detail is present.")
    else:
        score -= 6
        issues.append("Scope or use case detail is not clearly stated.")

    if evidence["dependencies"]["found"]:
        strengths.append("Dependencies are documented or mentioned.")
    else:
        score -= 10
        issues.append("Dependencies are not documented.")

    acceptance_found = evidence["acceptance"]["found"]
    dor_found = evidence["dor_dod"]["dor_found"]
    dod_found = evidence["dor_dod"]["dod_found"]
    success_found = evidence["success_criteria"]["found"]

    if acceptance_found or dor_found or dod_found or success_found:
        strengths.append("Acceptance or readiness/completion criteria are present.")
    else:
        score -= 20
        issues.append("Acceptance criteria, DoR, DoD, or success criteria are missing.")

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

    if priority:
        strengths.append(f"Priority is set to {priority}.")
    else:
        score -= 3
        issues.append("Priority is missing.")

    if evidence["risk"]["found"]:
        strengths.append("Risk, safety, or mitigation content is present.")

    maturity_bonus = 0
    if dor_found:
        maturity_bonus += 4
    if dod_found:
        maturity_bonus += 4
    if success_found:
        maturity_bonus += 3
    if evidence["business_context"]["found"] and evidence["scope"]["found"]:
        maturity_bonus += 2

    score += maturity_bonus
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

    suggested_rewrite = f"""Business need:
{summary or "State the business objective clearly."}

Problem statement:
Describe the current operational or delivery problem, including why this work matters.

Scope:
Define the in-scope use cases, systems, and expected outcomes. State any out-of-scope items.

Dependencies:
- List upstream/downstream systems
- List required approvals, environments, vendors, and access requirements

Success Criteria:
- Define measurable outcomes and thresholds for success
- State how results will be validated

Definition of Ready:
- Required systems, documents, owners, and dependencies are identified
- Scope and success criteria are agreed

Definition of Done:
- Solution implemented and validated
- Evidence captured
- Stakeholder sign-off recorded
"""

    return {
        "score": score,
        "status": status,
        "compliance": compliance,
        "issues": issues,
        "strengths": strengths,
        "evidence": evidence,
        "suggested_rewrite": suggested_rewrite.strip(),
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

    response = requests.get(url, headers=jira_auth_headers(), params=params, timeout=60)
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
    response = requests.get(url, headers=jira_auth_headers(), params=params, timeout=30)
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
    return evaluate_ticket(data)


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

    response = requests.get(url, headers=jira_auth_headers(), params=params, timeout=30)

    if response.status_code != 200:
        return JSONResponse(status_code=response.status_code, content=safe_json_response(response))

    issue = response.json()
    normalized = normalize_live_issue(issue)
    assessment = evaluate_ticket(normalized)

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

Problem statement:
{description or "Describe the current challenge, target systems, and expected outcome."}

Scope:
- Define in-scope use cases
- Define out-of-scope items if relevant

Dependencies:
- Systems, teams, vendors, approvals, and environment requirements

Success Criteria:
- Measurable improvements or validation thresholds
- Evidence required for successful completion

Definition of Ready:
- Scope agreed
- Dependencies identified
- Owners and environments confirmed

Definition of Done:
- Delivery completed
- Validation passed
- Evidence captured
- Stakeholder sign-off obtained
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
        return JSONResponse(status_code=400, content={"error": "summary is required"})

    url = f"{JIRA_BASE_URL}/rest/api/3/issue"
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": build_adf_document(description),
            "issuetype": {"name": issue_type},
        }
    }

    response = requests.post(url, headers=jira_auth_headers(), json=payload, timeout=30)
    result = safe_json_response(response)
    result["projectKey"] = project_key
    return result


@app.post("/api/jira/update")
def jira_update(data: Dict[str, Any]):
    issue_key = (data.get("issueKey") or "").strip().upper()
    summary = data.get("summary")
    description = data.get("description")
    issue_type = data.get("issueType")

    if not issue_key:
        return JSONResponse(status_code=400, content={"error": "issueKey is required"})

    fields_to_update: Dict[str, Any] = {}

    if summary is not None and str(summary).strip():
        fields_to_update["summary"] = str(summary).strip()

    if description is not None:
        fields_to_update["description"] = build_adf_document(str(description))

    if issue_type is not None and str(issue_type).strip():
        fields_to_update["issuetype"] = {"name": str(issue_type).strip()}

    if not fields_to_update:
        return JSONResponse(
            status_code=400,
            content={"error": "At least one of summary, description, or issueType must be provided"}
        )

    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}"
    payload = {"fields": fields_to_update}

    response = requests.put(url, headers=jira_auth_headers(), json=payload, timeout=30)

    result = {
        "status_code": response.status_code,
        "issueKey": issue_key,
    }

    if response.text.strip():
        try:
            result["jira_response"] = response.json()
        except Exception:
            result["jira_response"] = {"raw_text": response.text}
    else:
        result["jira_response"] = {"message": "Issue updated successfully"}

    return result