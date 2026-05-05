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

app = FastAPI(title="Jira Ticket Quality Compliance AI Dashboard")

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


def safe_json_response(response: requests.Response) -> Dict[str, Any]:
    try:
        body = response.json()
    except Exception:
        body = {"raw_text": response.text}
    return {
        "status_code": response.status_code,
        "jira_response": body,
    }


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


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def first_nonempty(items: List[str]) -> str:
    for item in items:
        if item and item.strip():
            return item.strip()
    return ""


def dedupe_keep_order(items: List[str], limit: int = 10) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        clean = re.sub(r"\s+", " ", str(item).strip())
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            out.append(clean)
        if len(out) >= limit:
            break
    return out


def find_sentences_with_keywords(text: str, keywords: List[str], max_hits: int = 8) -> List[str]:
    hits: List[str] = []
    sentences = split_sentences(text)

    for sentence in sentences:
        lower = sentence.lower()
        if any(keyword.lower() in lower for keyword in keywords):
            hits.append(sentence)
        if len(hits) >= max_hits:
            break

    return dedupe_keep_order(hits, max_hits)


def find_section_snippet(text: str, headings: List[str], window: int = 1800) -> str:
    lower = text.lower()
    for heading in headings:
        idx = lower.find(heading.lower())
        if idx != -1:
            return text[idx: idx + window].strip()
    return ""


def extract_bullets_from_text(text: str, max_items: int = 10) -> List[str]:
    candidates = re.split(r"(?:\n|•|- |\* )", text)
    cleaned = []
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip(" :-\t\r\n")
        if len(c) > 15:
            cleaned.append(c)
    return dedupe_keep_order(cleaned, max_items)


def detect_business_context(description: str) -> Dict[str, Any]:
    section = find_section_snippet(description, ["initiative summary", "overview", "business problem", "business need", "strategic value"])
    hits = find_sentences_with_keywords(
        description,
        ["business problem", "business need", "overview", "initiative summary", "operational impact", "strategic value", "results in", "outcome"]
    )
    if section:
        hits.insert(0, section)
    return {"found": bool(hits), "evidence": dedupe_keep_order(hits, 8)}


def detect_scope(description: str) -> Dict[str, Any]:
    section = find_section_snippet(description, ["scope", "poc scope and capabilities", "scope & use cases", "use case", "use cases"])
    hits = find_sentences_with_keywords(
        description,
        ["scope", "use case", "use cases", "capabilities", "implements", "demonstrates", "out of scope", "in scope"]
    )
    if section:
        hits.insert(0, section)
    return {"found": bool(hits), "evidence": dedupe_keep_order(hits, 8)}


def detect_dependencies(description: str, explicit_dependencies: Any = None) -> Dict[str, Any]:
    evidence = []
    methods = []

    if isinstance(explicit_dependencies, list):
        joined = " ".join(str(x) for x in explicit_dependencies if x)
        if joined.strip():
            evidence.append(joined.strip())
            methods.append("explicit_dependencies_field")
    elif explicit_dependencies:
        evidence.append(str(explicit_dependencies).strip())
        methods.append("explicit_dependencies_field")

    section = find_section_snippet(description, ["inputs & dependencies", "dependencies"])
    if section:
        evidence.append(section)
        methods.append("dependency_section")

    hits = find_sentences_with_keywords(
        description,
        ["dependency", "dependencies", "depends on", "required environments", "access and credentials", "vendor dependencies", "approvals", "approval", "security", "firewall", "access"]
    )
    if hits:
        evidence.extend(hits)
        methods.append("dependency_mentions")

    return {
        "found": bool(evidence),
        "methods": dedupe_keep_order(methods, 5),
        "evidence": dedupe_keep_order(evidence, 8)
    }


def detect_acceptance(description: str, explicit_ac: Any = None) -> Dict[str, Any]:
    evidence = []
    methods = []

    if isinstance(explicit_ac, list):
        joined = " ".join(str(x) for x in explicit_ac if x)
        if joined.strip():
            evidence.append(joined.strip())
            methods.append("explicit_acceptance_criteria_field")
    elif explicit_ac:
        evidence.append(str(explicit_ac).strip())
        methods.append("explicit_acceptance_criteria_field")

    lower = description.lower()

    if all(token in lower for token in ["given", "when", "then"]):
        evidence.extend(find_sentences_with_keywords(description, ["given", "when", "then"]))
        methods.append("given_when_then")

    for headings, method in [
        (["definition of ready", "dor"], "definition_of_ready"),
        (["definition of done", "dod"], "definition_of_done"),
        (["success criteria"], "success_criteria"),
        (["acceptance criteria"], "acceptance_criteria_heading"),
        (["ready when"], "ready_when"),
        (["done when"], "done_when"),
    ]:
        section = find_section_snippet(description, headings)
        if section:
            evidence.append(section)
            methods.append(method)

    return {
        "found": bool(evidence),
        "methods": dedupe_keep_order(methods, 8),
        "evidence": dedupe_keep_order(evidence, 8)
    }


def detect_success_criteria(description: str) -> Dict[str, Any]:
    section = find_section_snippet(description, ["success criteria", "successful if", "success criteria & measurement"])
    hits = find_sentences_with_keywords(
        description,
        ["success criteria", "successful if", "measured", "reduction", "mttr", "effort", "threshold", "go/no-go", "improved"]
    )
    if section:
        hits.insert(0, section)
    return {"found": bool(hits), "evidence": dedupe_keep_order(hits, 8)}


def detect_risk(description: str) -> Dict[str, Any]:
    section = find_section_snippet(description, ["non-functional & risk", "quality & safety", "risks", "risk"])
    hits = find_sentences_with_keywords(
        description,
        ["risk", "risks", "mitigation", "rollback", "constraints", "security", "safe", "guardrails", "negative-path"]
    )
    if section:
        hits.insert(0, section)
    return {"found": bool(hits), "evidence": dedupe_keep_order(hits, 8)}


def detect_dor_dod(description: str) -> Dict[str, Any]:
    dor = find_section_snippet(description, ["definition of ready", "dor"])
    dod = find_section_snippet(description, ["definition of done", "dod"])
    return {
        "dor_found": bool(dor),
        "dod_found": bool(dod),
        "dor_evidence": [dor] if dor else [],
        "dod_evidence": [dod] if dod else [],
    }


def guess_problem_statement(summary: str, description: str) -> str:
    candidates = []
    section = find_section_snippet(description, ["business problem", "problem statement"])
    if section:
        candidates.append(section)

    hits = find_sentences_with_keywords(
        description,
        ["manual", "tribal knowledge", "risk", "mttr", "inconsistent", "dependency", "slow", "current", "results in", "problem"]
    )
    candidates.extend(hits)

    if candidates:
        return dedupe_keep_order(candidates, 3)[0]

    if summary:
        return f"The current delivery or operational process related to '{summary}' needs a clearer, safer, and more repeatable execution path."
    return "The current process requires clearer business context, execution detail, and measurable delivery outcomes."


def infer_scope_points(summary: str, description: str) -> List[str]:
    section = find_section_snippet(description, ["scope", "poc scope and capabilities", "scope & use cases"])
    points = []

    if section:
        points.extend(extract_bullets_from_text(section, 8))

    hits = find_sentences_with_keywords(
        description,
        ["use case", "use cases", "implements", "demonstrates", "workflow", "execution", "integration", "validation"]
    )
    points.extend(hits)

    if not points and summary:
        points.append(f"Deliver the work required for {summary}.")
        points.append("Define the target systems, workflow, and expected operational outcome.")
        points.append("Clarify what is in scope and what is out of scope for this ticket.")

    return dedupe_keep_order(points, 6)


def infer_dependency_points(summary: str, description: str) -> List[str]:
    dep = detect_dependencies(description)
    points = []
    points.extend(dep["evidence"])

    if not points:
        if "firewall" in summary.lower():
            points.append("Firewall rule approval and network security validation are required.")
            points.append("Source, destination, port, and protocol details must be confirmed before implementation.")
        if "access" in summary.lower():
            points.append("Required system access, credentials, and approvals must be available before execution.")

    if not points:
        points.append("Required systems, approvals, credentials, and stakeholder inputs must be identified before execution.")
        points.append("Any upstream or downstream system dependency must be documented and confirmed.")

    return dedupe_keep_order(points, 6)


def infer_success_points(summary: str, description: str) -> List[str]:
    success = detect_success_criteria(description)
    points = []
    points.extend(success["evidence"])

    if not points:
        if "firewall" in summary.lower():
            points.append("The change is successful when connectivity is permitted only for the approved source/destination path.")
            points.append("Validation evidence confirms that the target service is reachable and policy-compliant.")
        else:
            points.append("The work is successful when the agreed business outcome is achieved and evidence is recorded.")
            points.append("Execution results must be repeatable, auditable, and validated by stakeholders.")

    return dedupe_keep_order(points, 5)


def infer_acceptance_criteria(summary: str, description: str) -> List[str]:
    acceptance = detect_acceptance(description)
    if acceptance["found"] and any("given" in e.lower() and "when" in e.lower() and "then" in e.lower() for e in acceptance["evidence"]):
        return dedupe_keep_order(acceptance["evidence"], 5)

    summary_lower = summary.lower()

    if "firewall" in summary_lower:
        return [
            "Given approved source and destination details, when the firewall rule is implemented, then the permitted application servers can reach the approved Instana SaaS endpoints successfully.",
            "Given an invalid or incomplete access request, when implementation is attempted, then the change is stopped and the missing prerequisite is identified.",
            "Given the rule is implemented, when validation is performed, then connectivity evidence and security approval are recorded."
        ]

    if "access" in summary_lower:
        return [
            "Given the required approvals and credentials are available, when access is configured, then the intended users or systems can access the required platform successfully.",
            "Given missing or invalid prerequisites, when the change is attempted, then the process is halted and the gap is reported.",
            "Given the change is complete, when validation is performed, then access works as expected and evidence is retained."
        ]

    if "poc" in summary_lower or "proof of concept" in description.lower():
        return [
            "Given the agreed scope and environments are available, when the POC is executed, then the target use cases run successfully end to end.",
            "Given the solution output is produced, when results are reviewed, then measurable evidence of effort reduction, consistency, or improved turnaround is available.",
            "Given the POC completes, when stakeholders assess the outcome, then a clear go/no-go or refine recommendation is documented."
        ]

    return [
        "Given the required prerequisites are available, when the work is executed, then the intended technical outcome is achieved successfully.",
        "Given missing or invalid prerequisites, when execution is attempted, then the issue is rejected or controlled safely.",
        "Given implementation is complete, when validation is performed, then evidence confirms the expected result and stakeholder sign-off can proceed."
    ]


def infer_dor_points(summary: str, description: str) -> List[str]:
    dor = detect_dor_dod(description)
    points = []

    if dor["dor_found"]:
        points.extend(extract_bullets_from_text(" ".join(dor["dor_evidence"]), 8))

    if not points:
        points = [
            "Scope and intended outcome are agreed.",
            "Required systems, access, approvals, and dependencies are identified.",
            "Relevant stakeholders, owners, and validation approach are confirmed."
        ]

    return dedupe_keep_order(points, 6)


def infer_dod_points(summary: str, description: str) -> List[str]:
    dod = detect_dor_dod(description)
    points = []

    if dod["dod_found"]:
        points.extend(extract_bullets_from_text(" ".join(dod["dod_evidence"]), 8))

    if not points:
        points = [
            "Implementation is completed and validated.",
            "Evidence of successful execution is captured.",
            "Known issues, constraints, and stakeholder sign-off are documented."
        ]

    return dedupe_keep_order(points, 6)


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
        "dependencies": detect_dependencies(description, dependencies),
        "acceptance": detect_acceptance(description, acceptance_criteria),
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

    return {
        "score": score,
        "status": status,
        "compliance": compliance,
        "issues": issues,
        "strengths": strengths,
        "evidence": evidence,
    }


def generate_smart_rewrite(summary: str, description: str, issue_type: str = "Story") -> Dict[str, Any]:
    business_context = detect_business_context(description)
    dependencies = infer_dependency_points(summary, description)
    success = infer_success_points(summary, description)
    acceptance = infer_acceptance_criteria(summary, description)
    dor = infer_dor_points(summary, description)
    dod = infer_dod_points(summary, description)
    risks = detect_risk(description)
    problem = guess_problem_statement(summary, description)
    scope_points = infer_scope_points(summary, description)

    business_text = first_nonempty(business_context["evidence"]) or f"This work item addresses the need for {summary}."
    risk_points = risks["evidence"] if risks["evidence"] else [
        "Execution must be controlled safely, with required approvals and validation evidence retained."
    ]

    rewritten = f"""Business need:
{summary}

Context:
{business_text}

Problem statement:
{problem}

Scope:
{chr(10).join(f"- {p}" for p in scope_points[:5])}

Dependencies:
{chr(10).join(f"- {p}" for p in dependencies[:5])}

Success Criteria:
{chr(10).join(f"- {p}" for p in success[:4])}

Acceptance Criteria:
{chr(10).join(f"- {p}" for p in acceptance[:4])}

Definition of Ready:
{chr(10).join(f"- {p}" for p in dor[:5])}

Definition of Done:
{chr(10).join(f"- {p}" for p in dod[:5])}

Risks / Controls:
{chr(10).join(f"- {p}" for p in risk_points[:4])}
""".strip()

    return {
        "summary": summary,
        "issue_type": issue_type,
        "rewritten_description": rewritten,
        "generated_sections": {
            "business_need": business_text,
            "problem_statement": problem,
            "scope": scope_points,
            "dependencies": dependencies,
            "success_criteria": success,
            "acceptance_criteria": acceptance,
            "definition_of_ready": dor,
            "definition_of_done": dod,
            "risks": risk_points,
        }
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


def summarize_bulk_results(project_key: str, jql: str, assessed_tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(assessed_tickets)
    if total == 0:
        return {
            "projectKey": project_key,
            "jql": jql,
            "summary": {
                "total_tickets": 0,
                "average_score": 0,
                "pass_count": 0,
                "partial_count": 0,
                "fail_count": 0,
                "missing_description_count": 0,
                "missing_labels_count": 0,
                "missing_components_count": 0,
                "missing_acceptance_count": 0,
            },
            "tickets": []
        }

    avg_score = round(sum(t["assessment"]["score"] for t in assessed_tickets) / total, 1)
    pass_count = sum(1 for t in assessed_tickets if t["assessment"]["status"] == "PASS")
    partial_count = sum(1 for t in assessed_tickets if t["assessment"]["status"] == "PARTIAL")
    fail_count = sum(1 for t in assessed_tickets if t["assessment"]["status"] == "FAIL")

    missing_description_count = sum(1 for t in assessed_tickets if not (t["normalized_issue"]["description"] or "").strip())
    missing_labels_count = sum(1 for t in assessed_tickets if len(t["normalized_issue"]["labels"]) == 0)
    missing_components_count = sum(1 for t in assessed_tickets if len(t["normalized_issue"]["components"]) == 0)
    missing_acceptance_count = sum(
        1 for t in assessed_tickets
        if not (
            t["assessment"]["evidence"]["acceptance"]["found"]
            or t["assessment"]["evidence"]["dor_dod"]["dor_found"]
            or t["assessment"]["evidence"]["dor_dod"]["dod_found"]
            or t["assessment"]["evidence"]["success_criteria"]["found"]
        )
    )

    sorted_tickets = sorted(
        assessed_tickets,
        key=lambda x: (x["assessment"]["score"], x["normalized_issue"]["issue_key"])
    )

    return {
        "projectKey": project_key,
        "jql": jql,
        "summary": {
            "total_tickets": total,
            "average_score": avg_score,
            "pass_count": pass_count,
            "partial_count": partial_count,
            "fail_count": fail_count,
            "missing_description_count": missing_description_count,
            "missing_labels_count": missing_labels_count,
            "missing_components_count": missing_components_count,
            "missing_acceptance_count": missing_acceptance_count,
        },
        "tickets": sorted_tickets
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "default_project": DEFAULT_PROJECT})


@app.get("/api/health")
def api_health():
    return {"ok": True, "jira_base_url": JIRA_BASE_URL, "default_project": DEFAULT_PROJECT}


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
            "summary", "description", "status", "assignee", "issuetype",
            "labels", "priority", "components", "created", "updated", "reporter"
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
        return JSONResponse(status_code=400, content={"error": "issueKey is required, for example ARC-827 or NMGOS-3846"})

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
    issue_type = (payload.get("issueType") or payload.get("issue_type") or "Story").strip()

    return generate_smart_rewrite(summary, description, issue_type)


@app.post("/api/tickets/rewrite-live")
def rewrite_live_ticket(payload: Dict[str, Any]):
    issue_key = (payload.get("issueKey") or "").strip().upper()
    if not issue_key:
        return JSONResponse(status_code=400, content={"error": "issueKey is required"})

    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}"
    params = {
        "fields": "summary,description,status,assignee,issuetype,labels,priority,components,created,updated,reporter"
    }

    response = requests.get(url, headers=jira_auth_headers(), params=params, timeout=30)
    if response.status_code != 200:
        return JSONResponse(status_code=response.status_code, content=safe_json_response(response))

    issue = response.json()
    normalized = normalize_live_issue(issue)
    rewrite = generate_smart_rewrite(
        normalized["summary"],
        normalized["description"],
        normalized["issue_type"] or "Story"
    )

    return {
        "issueKey": issue_key,
        "normalized_issue": normalized,
        "rewrite": rewrite
    }


@app.post("/api/tickets/bulk-dashboard")
def bulk_dashboard(payload: Dict[str, Any]):
    project_key = normalize_project_key(payload.get("projectKey"))
    jql = payload.get("jql") or f"project = {project_key} ORDER BY created DESC"
    max_results = int(payload.get("maxResults", 25))
    fields = [
        "summary", "description", "status", "assignee", "issuetype",
        "labels", "priority", "components", "created", "updated", "reporter"
    ]

    url = f"{JIRA_BASE_URL}/rest/api/3/search/jql"
    params = {
        "jql": jql,
        "maxResults": max_results,
        "fields": ",".join(fields),
        "fieldsByKeys": "false",
    }

    response = requests.get(url, headers=jira_auth_headers(), params=params, timeout=90)
    if response.status_code != 200:
        return JSONResponse(status_code=response.status_code, content=safe_json_response(response))

    body = response.json()
    issues = body.get("issues", [])

    assessed_tickets = []
    for issue in issues:
        normalized = normalize_live_issue(issue)
        assessment = evaluate_ticket(normalized)
        assessed_tickets.append({
            "issueKey": normalized["issue_key"],
            "normalized_issue": normalized,
            "assessment": assessment,
        })

    result = summarize_bulk_results(project_key, jql, assessed_tickets)
    result["nextPageToken"] = body.get("nextPageToken")
    result["isLast"] = body.get("isLast")
    return result


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
        return JSONResponse(status_code=400, content={"error": "At least one of summary, description, or issueType must be provided"})

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