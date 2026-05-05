import os
import re
import csv
import io
import base64
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

load_dotenv()

app = FastAPI(title="Jira Ticket Quality Compliance AI")
templates = Jinja2Templates(directory="templates")

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "").rstrip("/")
JIRA_EMAIL = os.getenv("JIRA_EMAIL", "")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")
DEFAULT_PROJECT = os.getenv("JIRA_PROJECT_KEY", "ARC").strip().upper()
STORY_POINTS_FIELD_ID = os.getenv("JIRA_STORY_POINTS_FIELD", "customfield_10016")
DEFAULT_COMPONENTS = [x.strip() for x in os.getenv("DEFAULT_COMPONENTS", "").split(",") if x.strip()]
DEFAULT_LABELS = [x.strip() for x in os.getenv("DEFAULT_LABELS", "").split(",") if x.strip()]

SECTION_ORDER = [
    "Business Need", "Context", "Problem Statement", "Scope", "Dependencies",
    "Success Criteria", "Acceptance Criteria", "Definition of Ready",
    "Definition of Done", "Risks / Controls", "Additional Notes",
]
SECTION_PATTERNS = {
    "Business Need": [r"business need\s*:"],
    "Context": [r"context\s*:"],
    "Problem Statement": [r"problem statement\s*:"],
    "Scope": [r"scope\s*:"],
    "Dependencies": [r"dependencies\s*:"],
    "Success Criteria": [r"success criteria\s*:"],
    "Acceptance Criteria": [r"acceptance criteria\s*:"],
    "Definition of Ready": [r"definition of ready\s*:", r"\bdor\b"],
    "Definition of Done": [r"definition of done\s*:", r"\bdod\b"],
    "Risks / Controls": [r"risks?\s*/\s*controls\s*:", r"risks\s*:"],
    "Additional Notes": [r"additional notes\s*:"]
}


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
    return {"status_code": response.status_code, "jira_response": body}


def normalize_to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(x).strip() for x in value if str(x).strip()]
    elif isinstance(value, str):
        parts = re.split(r"\n|,|;|•|- ", value)
        items = [p.strip() for p in parts if p.strip()]
    else:
        items = [str(value).strip()]
    return dedupe_keep_order(items, 20)


def dedupe_keep_order(items: List[str], limit: int = 50) -> List[str]:
    out, seen = [], set()
    for item in items:
        clean = re.sub(r"\s+", " ", str(item).strip())
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            out.append(clean)
        if len(out) >= limit:
            break
    return out


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def extract_bullets_from_text(text: str, max_items: int = 10) -> List[str]:
    candidates = re.split(r"(?:\n|•|- |\* )", text)
    cleaned = []
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip(" :-\t\r\n")
        if len(c) > 8:
            cleaned.append(c)
    return dedupe_keep_order(cleaned, max_items)


def first_nonempty(items: List[str]) -> str:
    for item in items:
        if item and item.strip():
            return item.strip()
    return ""


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
    return re.sub(r"\s+", " ", " ".join(t.strip() for t in texts if str(t).strip())).strip()


def build_adf_document(text: str) -> Dict[str, Any]:
    paragraphs = [p.strip() for p in (text or "").split("\n") if p.strip()] or [""]
    return {
        "type": "doc",
        "version": 1,
        "content": [{"type": "paragraph", "content": [{"type": "text", "text": para}]} for para in paragraphs],
    }


def find_section_snippet(text: str, headings: List[str], window: int = 1800) -> str:
    lower = text.lower()
    for heading in headings:
        idx = lower.find(heading.lower())
        if idx != -1:
            return text[idx: idx + window].strip()
    return ""


def find_sentences_with_keywords(text: str, keywords: List[str], max_hits: int = 8) -> List[str]:
    hits = []
    for sentence in split_sentences(text):
        lower = sentence.lower()
        if any(k.lower() in lower for k in keywords):
            hits.append(sentence)
        if len(hits) >= max_hits:
            break
    return dedupe_keep_order(hits, max_hits)


def detect_business_context(description: str) -> Dict[str, Any]:
    section = find_section_snippet(description, ["initiative summary", "overview", "business problem", "business need", "strategic value"])
    hits = find_sentences_with_keywords(description, ["business problem", "business need", "overview", "initiative summary", "outcome"])
    if section:
        hits.insert(0, section)
    return {"found": bool(hits), "evidence": dedupe_keep_order(hits, 8)}


def detect_scope(description: str) -> Dict[str, Any]:
    section = find_section_snippet(description, ["scope", "use case", "use cases", "capabilities"])
    hits = find_sentences_with_keywords(description, ["scope", "use case", "capabilities", "workflow", "integration"])
    if section:
        hits.insert(0, section)
    return {"found": bool(hits), "evidence": dedupe_keep_order(hits, 8)}


def detect_dependencies(description: str, explicit_dependencies: Any = None) -> Dict[str, Any]:
    evidence, methods = [], []
    explicit = normalize_to_list(explicit_dependencies)
    if explicit:
        evidence.extend(explicit)
        methods.append("explicit_dependencies_field")
    section = find_section_snippet(description, ["dependencies", "inputs & dependencies"])
    if section:
        evidence.append(section)
        methods.append("dependency_section")
    hits = find_sentences_with_keywords(description, ["dependency", "dependencies", "approval", "approvals", "access", "credentials", "security"])
    if hits:
        evidence.extend(hits)
        methods.append("dependency_mentions")
    return {"found": bool(evidence), "methods": dedupe_keep_order(methods, 5), "evidence": dedupe_keep_order(evidence, 8)}


def detect_acceptance(description: str, explicit_ac: Any = None) -> Dict[str, Any]:
    evidence, methods = [], []
    explicit = normalize_to_list(explicit_ac)
    if explicit:
        evidence.extend(explicit)
        methods.append("explicit_acceptance_criteria_field")
    lower = description.lower()
    if all(token in lower for token in ["given", "when", "then"]):
        evidence.extend(find_sentences_with_keywords(description, ["given", "when", "then"]))
        methods.append("given_when_then")
    for headings, method in [(["definition of ready", "dor"], "definition_of_ready"), (["definition of done", "dod"], "definition_of_done"), (["success criteria"], "success_criteria"), (["acceptance criteria"], "acceptance_criteria_heading")]:
        section = find_section_snippet(description, headings)
        if section:
            evidence.append(section)
            methods.append(method)
    return {"found": bool(evidence), "methods": dedupe_keep_order(methods, 8), "evidence": dedupe_keep_order(evidence, 8)}


def detect_success_criteria(description: str) -> Dict[str, Any]:
    section = find_section_snippet(description, ["success criteria", "successful if"])
    hits = find_sentences_with_keywords(description, ["success criteria", "threshold", "validated", "measured"])
    if section:
        hits.insert(0, section)
    return {"found": bool(hits), "evidence": dedupe_keep_order(hits, 8)}


def detect_risk(description: str) -> Dict[str, Any]:
    section = find_section_snippet(description, ["risks / controls", "risk", "risks"])
    hits = find_sentences_with_keywords(description, ["risk", "risks", "mitigation", "controls", "safe", "guardrails"])
    if section:
        hits.insert(0, section)
    return {"found": bool(hits), "evidence": dedupe_keep_order(hits, 8)}


def detect_dor_dod(description: str) -> Dict[str, Any]:
    dor = find_section_snippet(description, ["definition of ready", "dor"])
    dod = find_section_snippet(description, ["definition of done", "dod"])
    return {"dor_found": bool(dor), "dod_found": bool(dod), "dor_evidence": [dor] if dor else [], "dod_evidence": [dod] if dod else []}


def guess_problem_statement(summary: str, description: str) -> str:
    section = find_section_snippet(description, ["problem statement", "business problem"])
    if section:
        return section.split("\n")[0].strip()
    hits = find_sentences_with_keywords(description, ["manual", "slow", "inconsistent", "problem", "delay", "effort"])
    if hits:
        return hits[0]
    if summary:
        return f"The current delivery or operational process related to '{summary}' needs a clearer, safer, and more repeatable execution path."
    return "The current process requires clearer business context, execution detail, and measurable delivery outcomes."


def infer_scope_points(summary: str, description: str) -> List[str]:
    section = find_section_snippet(description, ["scope", "use case", "capabilities"])
    points = extract_bullets_from_text(section, 8) if section else []
    if not points:
        points.extend(find_sentences_with_keywords(description, ["workflow", "integration", "deliver", "implement"], 6))
    if not points and summary:
        points = [f"Deliver the work required for {summary}.", "Define the target systems, workflow, and expected operational outcome.", "Clarify what is in scope and what is out of scope for this ticket."]
    return dedupe_keep_order(points, 6)


def infer_dependency_points(summary: str, description: str) -> List[str]:
    points = detect_dependencies(description)["evidence"]
    if not points:
        points = [
            "Required systems, approvals, credentials, and stakeholder inputs must be identified before execution.",
            "Any upstream or downstream system dependency must be documented and confirmed.",
        ]
    return dedupe_keep_order(points, 6)


def infer_success_points(summary: str, description: str) -> List[str]:
    points = detect_success_criteria(description)["evidence"]
    if not points:
        points = [
            "The work is successful when the agreed business outcome is achieved and evidence is recorded.",
            "Execution results must be repeatable, auditable, and validated by stakeholders.",
        ]
    return dedupe_keep_order(points, 5)


def infer_acceptance_criteria(summary: str, description: str) -> List[str]:
    acceptance = detect_acceptance(description)
    if acceptance["found"]:
        bullets = []
        for e in acceptance["evidence"]:
            bullets.extend(extract_bullets_from_text(e, 12))
        if bullets:
            return dedupe_keep_order(bullets, 5)
    return [
        "Given the required prerequisites are available, when the work is executed, then the intended technical outcome is achieved successfully.",
        "Given missing or invalid prerequisites, when execution is attempted, then the issue is rejected or controlled safely.",
        "Given implementation is complete, when validation is performed, then evidence confirms the expected result and stakeholder sign-off can proceed.",
    ]


def infer_dor_points(summary: str, description: str) -> List[str]:
    dor = detect_dor_dod(description)
    points = extract_bullets_from_text(" ".join(dor["dor_evidence"]), 8) if dor["dor_found"] else []
    if not points:
        points = [
            "Scope and intended outcome are agreed.",
            "Required systems, access, approvals, and dependencies are identified.",
            "Relevant stakeholders, owners, and validation approach are confirmed.",
        ]
    return dedupe_keep_order(points, 6)


def infer_dod_points(summary: str, description: str) -> List[str]:
    dod = detect_dor_dod(description)
    points = extract_bullets_from_text(" ".join(dod["dod_evidence"]), 8) if dod["dod_found"] else []
    if not points:
        points = [
            "Implementation is completed and validated.",
            "Evidence of successful execution is captured.",
            "Known issues, constraints, and stakeholder sign-off are documented.",
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
        score -= 20; issues.append("Summary is too short.")
    elif len(summary) < 18:
        score -= 8; issues.append("Summary is present but not very specific.")
    else:
        strengths.append("Summary is reasonably specific.")
    if len(description) == 0:
        score -= 35; issues.append("Description is missing.")
    elif len(description) < 40:
        score -= 20; issues.append("Description is too short.")
    elif len(description) < 120:
        score -= 8; issues.append("Description exists but could use more delivery detail.")
    else:
        strengths.append("Description provides usable detail.")
    if not issue_type:
        score -= 5; issues.append("Issue type is missing.")
    else:
        strengths.append(f"Issue type is set to {issue_type}.")
    if evidence["business_context"]["found"]:
        strengths.append("Business context is present.")
    else:
        score -= 8; issues.append("Business context is not clearly stated.")
    if evidence["scope"]["found"]:
        strengths.append("Scope or use case detail is present.")
    else:
        score -= 6; issues.append("Scope or use case detail is not clearly stated.")
    if evidence["dependencies"]["found"]:
        strengths.append("Dependencies are documented or mentioned.")
    else:
        score -= 10; issues.append("Dependencies are not documented.")
    if evidence["acceptance"]["found"] or evidence["dor_dod"]["dor_found"] or evidence["dor_dod"]["dod_found"] or evidence["success_criteria"]["found"]:
        strengths.append("Acceptance or readiness/completion criteria are present.")
    else:
        score -= 20; issues.append("Acceptance criteria, DoR, DoD, or success criteria are missing.")
    if labels:
        strengths.append("Labels are present.")
    else:
        score -= 5; issues.append("Labels are missing.")
    if components:
        strengths.append("Components are present.")
    else:
        score -= 5; issues.append("Components are missing.")
    if priority:
        strengths.append(f"Priority is set to {priority}.")
    else:
        score -= 3; issues.append("Priority is missing.")
    if evidence["risk"]["found"]:
        strengths.append("Risk, safety, or mitigation content is present.")

    maturity_bonus = 0
    if evidence["dor_dod"]["dor_found"]: maturity_bonus += 4
    if evidence["dor_dod"]["dod_found"]: maturity_bonus += 4
    if evidence["success_criteria"]["found"]: maturity_bonus += 3
    if evidence["business_context"]["found"] and evidence["scope"]["found"]: maturity_bonus += 2
    score = max(0, min(score + maturity_bonus, 100))
    if score >= 80:
        status, compliance = "PASS", "Strong"
    elif score >= 60:
        status, compliance = "PARTIAL", "Moderate"
    else:
        status, compliance = "FAIL", "Weak"
    return {"score": score, "status": status, "compliance": compliance, "issues": issues, "strengths": strengths, "evidence": evidence}


def clean_text_noise(text: str) -> str:
    if not text:
        return ""
    replacements = [
        (r"\bDependencies:\s*-\s*Dependencies:\s*", "Dependencies:\n- "),
        (r"\bSuccess Criteria:\s*-\s*Success Criteria:\s*", "Success Criteria:\n- "),
        (r"\bAcceptance Criteria:\s*-\s*Acceptance Criteria:\s*", "Acceptance Criteria:\n- "),
        (r"\bDefinition of Ready:\s*-\s*Definition of Ready\b", "Definition of Ready:"),
        (r"\bDefinition of Done:\s*-\s*Definition of Done\b", "Definition of Done:"),
        (r"\bRisks?\s*/\s*Controls:\s*-\s*Risks?\s*/\s*Controls:\s*", "Risks / Controls:\n- "),
        (r"-\s*-\s*", "- "),
        (r"\s+\n", "\n"),
        (r"\n{3,}", "\n\n"),
        (r"[ \t]{2,}", " "),
    ]
    out = text
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out.strip()


def split_into_sections(raw_text: str) -> Dict[str, List[str]]:
    text = clean_text_noise(raw_text)
    sections = {name: [] for name in SECTION_ORDER}
    heading_matches = []
    for section_name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                heading_matches.append((m.start(), m.end(), section_name))
    heading_matches.sort(key=lambda x: x[0])
    if not heading_matches:
        sections["Context"].append(text)
        return sections
    for idx, (start, end, section_name) in enumerate(heading_matches):
        next_start = heading_matches[idx + 1][0] if idx + 1 < len(heading_matches) else len(text)
        chunk = text[end:next_start].strip(" \n:-")
        if chunk:
            sections[section_name].append(chunk)
    return sections


def normalize_section_items(section_name: str, chunks: List[str]) -> List[str]:
    items: List[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        if section_name in {"Scope", "Dependencies", "Success Criteria", "Acceptance Criteria", "Definition of Ready", "Definition of Done", "Risks / Controls"}:
            extracted = extract_bullets_from_text(chunk, 20)
            items.extend(extracted if extracted else split_sentences(chunk))
        else:
            items.extend([chunk] if section_name == "Context" else split_sentences(chunk))
    cleaned = []
    for item in items:
        item = re.sub(r"^\s*[-•*]\s*", "", str(item)).strip()
        item = re.sub(r"\s+", " ", item).strip()
        if len(item) >= 6:
            cleaned.append(item)
    return dedupe_keep_order(cleaned, 12)


def choose_best_summary(summary: str, description: str) -> str:
    summary = (summary or "").strip()
    if len(summary) >= 20 and any(word in summary.lower() for word in ["platform", "automation", "evaluation", "engine", "system"]):
        return summary
    if "rfp" in summary.lower() or "evaluator" in summary.lower():
        return "AI-driven RFP Evaluation Platform to Automate Vendor Scoring and Reduce Technical Review Effort"
    if summary:
        return summary
    derived = first_nonempty(split_sentences(description))
    return derived[:120] if derived else "Normalized Jira Ticket"


def auto_clean_and_normalize_ticket(summary: str, description: str, issue_type: str = "Story") -> Dict[str, Any]:
    sections = split_into_sections(description or "")
    business_need = normalize_section_items("Business Need", sections["Business Need"])
    context = normalize_section_items("Context", sections["Context"])
    problem = normalize_section_items("Problem Statement", sections["Problem Statement"])
    scope = normalize_section_items("Scope", sections["Scope"])
    dependencies = normalize_section_items("Dependencies", sections["Dependencies"])
    success = normalize_section_items("Success Criteria", sections["Success Criteria"])
    acceptance = normalize_section_items("Acceptance Criteria", sections["Acceptance Criteria"])
    dor = normalize_section_items("Definition of Ready", sections["Definition of Ready"])
    dod = normalize_section_items("Definition of Done", sections["Definition of Done"])
    risks = normalize_section_items("Risks / Controls", sections["Risks / Controls"])
    notes = normalize_section_items("Additional Notes", sections["Additional Notes"])

    best_summary = choose_best_summary(summary, description)
    if not business_need:
        business_need = [best_summary]
    if not context:
        detected_context = detect_business_context(description)["evidence"]
        context = detected_context[:2] if detected_context else [f"This work item addresses the need for {best_summary}."]
    if not problem:
        problem = [guess_problem_statement(best_summary, description)]
    if not scope:
        scope = infer_scope_points(best_summary, description)
    if not dependencies:
        dependencies = infer_dependency_points(best_summary, description)
    if not success:
        success = infer_success_points(best_summary, description)
    if not acceptance:
        acceptance = infer_acceptance_criteria(best_summary, description)
    if not dor:
        dor = infer_dor_points(best_summary, description)
    if not dod:
        dod = infer_dod_points(best_summary, description)
    if not risks:
        risks = detect_risk(description)["evidence"] or ["Execution must be controlled safely, with required approvals and validation evidence retained."]

    normalized_description = f"""Business Need:
{business_need[0]}

Context:
{" ".join(context[:2])}

Problem Statement:
{problem[0]}

Scope:
{chr(10).join(f"- {x}" for x in dedupe_keep_order(scope, 6))}

Dependencies:
{chr(10).join(f"- {x}" for x in dedupe_keep_order(dependencies, 6))}

Success Criteria:
{chr(10).join(f"- {x}" for x in dedupe_keep_order(success, 5))}

Acceptance Criteria:
{chr(10).join(f"- {x}" for x in dedupe_keep_order(acceptance, 5))}

Definition of Ready:
{chr(10).join(f"- {x}" for x in dedupe_keep_order(dor, 6))}

Definition of Done:
{chr(10).join(f"- {x}" for x in dedupe_keep_order(dod, 6))}

Risks / Controls:
{chr(10).join(f"- {x}" for x in dedupe_keep_order(risks, 5))}
""".strip()
    if notes:
        normalized_description += f"\n\nAdditional Notes:\n" + "\n".join(f"- {x}" for x in dedupe_keep_order(notes, 5))
    return {"summary": best_summary, "issue_type": issue_type, "normalized_description": normalized_description}


def merge_human_input_into_rewrite(summary: str, existing_description: str, issue_type: str, human_input: Dict[str, Any]) -> Dict[str, Any]:
    business_objective = (human_input.get("business_objective") or "").strip()
    problem_statement = (human_input.get("problem_statement") or "").strip()
    systems_involved = normalize_to_list(human_input.get("systems_involved"))
    dependencies_input = normalize_to_list(human_input.get("dependencies"))
    risks_input = normalize_to_list(human_input.get("risks"))
    success_input = normalize_to_list(human_input.get("success_criteria"))
    validation_steps = normalize_to_list(human_input.get("validation_steps"))
    stakeholders = normalize_to_list(human_input.get("stakeholders"))
    approvals = normalize_to_list(human_input.get("approvals_needed"))
    additional_notes = (human_input.get("additional_notes") or "").strip()

    business_context = detect_business_context(existing_description)
    business_text = business_objective or first_nonempty(business_context["evidence"]) or f"This work item addresses the need for {summary}."
    problem_text = problem_statement or guess_problem_statement(summary, existing_description)
    scope_points = infer_scope_points(summary, existing_description)
    if systems_involved:
        scope_points = dedupe_keep_order(scope_points + [f"Involved system/domain: {x}" for x in systems_involved], 8)
    dependency_points = dedupe_keep_order(infer_dependency_points(summary, existing_description) + dependencies_input + approvals, 8)
    success_points = dedupe_keep_order(infer_success_points(summary, existing_description) + success_input + validation_steps, 8)
    acceptance_points = infer_acceptance_criteria(summary, existing_description)
    if validation_steps:
        acceptance_points = dedupe_keep_order(acceptance_points + [f"Given implementation is completed, when validation is performed, then {x}." for x in validation_steps[:2]], 6)
    dor_points = infer_dor_points(summary, existing_description)
    if stakeholders:
        dor_points = dedupe_keep_order(dor_points + [f"Stakeholder or owner identified: {x}" for x in stakeholders], 8)
    if approvals:
        dor_points = dedupe_keep_order(dor_points + [f"Required approval identified: {x}" for x in approvals], 8)
    dod_points = infer_dod_points(summary, existing_description)
    if validation_steps:
        dod_points = dedupe_keep_order(dod_points + [f"Validation completed: {x}" for x in validation_steps], 8)
    risk_points = dedupe_keep_order(detect_risk(existing_description)["evidence"] + risks_input, 8) or ["Execution must be controlled safely, with required approvals and validation evidence retained."]
    notes_block = f"\nAdditional notes:\n{additional_notes}" if additional_notes else ""

    rewritten = f"""Business Need:
{summary}

Context:
{business_text}

Problem Statement:
{problem_text}

Scope:
{chr(10).join(f"- {p}" for p in scope_points[:6])}

Dependencies:
{chr(10).join(f"- {p}" for p in dependency_points[:6])}

Success Criteria:
{chr(10).join(f"- {p}" for p in success_points[:5])}

Acceptance Criteria:
{chr(10).join(f"- {p}" for p in acceptance_points[:5])}

Definition of Ready:
{chr(10).join(f"- {p}" for p in dor_points[:6])}

Definition of Done:
{chr(10).join(f"- {p}" for p in dod_points[:6])}

Risks / Controls:
{chr(10).join(f"- {p}" for p in risk_points[:5])}{notes_block}
""".strip()
    return {"summary": summary, "issue_type": issue_type, "rewritten_description": rewritten}


def determine_child_issue_type(parent_issue_type: str) -> str:
    parent = (parent_issue_type or "").strip().lower()
    if parent == "initiative":
        return "Epic"
    if parent in {"epic", "feature"}:
        return "Story"
    if parent in {"story", "task", "bug"}:
        return "Sub-task"
    return "Story"


def estimate_story_points(text: str, complexity_inputs: Optional[List[str]] = None) -> int:
    text = (text or "").lower()
    factors = 0
    keywords = {"integration", "api", "multiple", "multi", "security", "workflow", "batch", "export", "dashboard", "parser", "llm", "ai", "scoring", "comparison", "evidence", "jira", "approval", "validation", "config", "upload", "document"}
    for keyword in keywords:
        if keyword in text:
            factors += 1
    if complexity_inputs:
        factors += len(complexity_inputs)
    if factors <= 2: return 2
    if factors <= 4: return 3
    if factors <= 7: return 5
    if factors <= 10: return 8
    return 13


def build_child_description(summary: str, parent_summary: str, detail_points: List[str], acceptance: List[str], dor: List[str], dod: List[str], risks: List[str]) -> str:
    return f"""Business Need:
{summary}

Context:
This child work item supports the parent ticket: {parent_summary}.

Problem Statement:
This work item is required to complete a meaningful part of the parent outcome in a structured and auditable way.

Scope:
{chr(10).join(f"- {x}" for x in dedupe_keep_order(detail_points, 5))}

Acceptance Criteria:
{chr(10).join(f"- {x}" for x in dedupe_keep_order(acceptance, 4))}

Definition of Ready:
{chr(10).join(f"- {x}" for x in dedupe_keep_order(dor, 4))}

Definition of Done:
{chr(10).join(f"- {x}" for x in dedupe_keep_order(dod, 4))}

Risks / Controls:
{chr(10).join(f"- {x}" for x in dedupe_keep_order(risks, 3))}
""".strip()


def suggest_breakdown(parent_issue: Dict[str, Any], mode: str = "standard") -> Dict[str, Any]:
    normalized = normalize_live_issue(parent_issue)
    parent_type = normalized["issue_type"] or ""
    parent_summary = normalized["summary"]
    child_type = determine_child_issue_type(parent_type)
    labels = dedupe_keep_order(DEFAULT_LABELS + normalized.get("labels", []), 10)
    components = dedupe_keep_order(DEFAULT_COMPONENTS + normalized.get("components", []), 10)

    if child_type == "Epic":
        themes = [
            ("Establish evaluation model and governance", ["Agree scoring model", "Define decision criteria", "Confirm governance and sign-off model"]),
            ("Deliver ingestion and document processing capability", ["Ingest RFP documents", "Parse vendor submissions", "Validate uploaded content"]),
            ("Build evaluation and comparison capability", ["Score vendors", "Compare responses", "Generate ranked insights"]),
            ("Deliver explainability and reporting capability", ["Evidence tracing", "Dashboard reporting", "CSV / PowerPoint export"]),
        ]
    elif child_type == "Story":
        themes = [
            ("Define evaluation model and scoring rules", ["Capture scoring matrix", "Model weighted criteria", "Validate scoring logic"]),
            ("Build document ingestion pipeline", ["Upload vendor files", "Extract content", "Normalize input formats"]),
            ("Implement vendor comparison engine", ["Compute comparative scores", "Highlight gaps", "Show strongest candidate evidence"]),
            ("Build explainability and evidence view", ["Trace output to source", "Present supporting evidence", "Improve reviewer trust"]),
            ("Add dashboard and export capability", ["Dashboard", "CSV export", "PowerPoint export"]),
        ]
    else:
        themes = [
            ("Design the implementation approach", ["Define technical approach", "Confirm fields, screens, and outputs"]),
            ("Implement backend logic", ["Create endpoint", "Process request", "Handle validation"]),
            ("Build UI interaction", ["Add interface controls", "Display results", "Support review workflow"]),
            ("Test and validate output", ["Run positive tests", "Run negative tests", "Capture evidence"]),
        ]

    if mode == "lean":
        themes = themes[:3]
    elif mode == "detailed":
        if child_type == "Story":
            themes.extend([
                ("Add admin configuration for templates and field mapping", ["Map project-specific fields", "Configure story point field", "Support labels/components defaults"]),
                ("Strengthen quality and performance", ["Optimize processing", "Improve consistency", "Harden error handling"]),
            ])
        elif child_type == "Epic":
            themes.append(("Establish delivery governance and rollout approach", ["Agree milestones", "Define ownership", "Plan release approach"]))

    suggested_children = []
    for idx, (theme, details) in enumerate(themes, start=1):
        acceptance = [
            f"Given the parent objective is approved, when this work item is completed, then the intended outcome for '{theme}' is delivered successfully.",
            "Given invalid or incomplete inputs, when execution is attempted, then the issue is handled safely and the gap is identified.",
            "Given implementation is complete, when validation is performed, then evidence confirms the expected result.",
        ]
        dor = ["Scope and intended outcome are agreed.", "Required systems, dependencies, and approvals are identified.", "The validation approach is defined."]
        dod = ["Implementation is completed and validated.", "Evidence of successful execution is captured.", "Any known issues or constraints are documented."]
        risks = ["Quality risk if scope is ambiguous.", "Execution risk if dependencies are not available.", "Control through validation and stakeholder review."]
        desc = build_child_description(theme, parent_summary, details, acceptance, dor, dod, risks)
        suggested_children.append({
            "issueType": child_type,
            "summary": theme,
            "description": desc,
            "labels": labels,
            "components": components,
            "priority": normalized.get("priority") or "Medium",
            "storyPoints": estimate_story_points(theme + " " + desc, details) if child_type == "Story" else None,
            "confidence": "High" if idx <= 2 else "Medium",
            "parentIssueKey": normalized["issue_key"],
        })

    return {
        "parentIssueKey": normalized["issue_key"],
        "parentIssueType": parent_type,
        "parentSummary": parent_summary,
        "suggestedChildIssueType": child_type,
        "mode": mode,
        "suggestedChildren": suggested_children,
        "notes": [
            "Initiative breaks down into Epics.",
            "Epic or Feature breaks down into Stories.",
            "Story breaks down into Sub-tasks.",
            "Story sizing is only suggested for Stories.",
        ],
    }


def create_single_issue(project_key: str, child: Dict[str, Any]) -> Dict[str, Any]:
    issue_type = child.get("issueType", "Story")
    fields: Dict[str, Any] = {
        "project": {"key": project_key},
        "summary": child.get("summary", "").strip(),
        "description": build_adf_document(child.get("description", "")),
        "issuetype": {"name": issue_type},
    }
    if child.get("labels"):
        fields["labels"] = normalize_to_list(child.get("labels"))
    if child.get("components"):
        fields["components"] = [{"name": x} for x in normalize_to_list(child.get("components"))]
    if child.get("priority"):
        fields["priority"] = {"name": child.get("priority")}
    if child.get("parentIssueKey"):
        fields["parent"] = {"key": child.get("parentIssueKey")}
    if issue_type.lower() == "story" and child.get("storyPoints") is not None:
        fields[STORY_POINTS_FIELD_ID] = child.get("storyPoints")
    response = requests.post(f"{JIRA_BASE_URL}/rest/api/3/issue", headers=jira_auth_headers(), json={"fields": fields}, timeout=30)
    result = safe_json_response(response)
    result["attempted_summary"] = child.get("summary")
    result["attempted_issue_type"] = issue_type
    return result


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "default_project": DEFAULT_PROJECT, "story_points_field": STORY_POINTS_FIELD_ID})


@app.get("/api/health")
def api_health():
    return {"ok": True, "jira_base_url": JIRA_BASE_URL, "default_project": DEFAULT_PROJECT, "story_points_field": STORY_POINTS_FIELD_ID}


@app.get("/api/jira/me")
def jira_me():
    response = requests.get(f"{JIRA_BASE_URL}/rest/api/3/myself", headers=jira_auth_headers(), timeout=30)
    return safe_json_response(response)


@app.post("/api/jira/search")
def jira_search(body: Dict[str, Any]):
    project_key = normalize_project_key(body.get("projectKey"))
    jql = body.get("jql") or f"project = {project_key} ORDER BY created DESC"
    max_results = int(body.get("maxResults", 10))
    next_page_token = body.get("nextPageToken")
    fields = body.get("fields", ["summary", "description", "status", "assignee", "issuetype", "labels", "priority", "components", "created", "updated", "reporter"])
    params = {"jql": jql, "maxResults": max_results, "fields": ",".join(fields), "fieldsByKeys": "false"}
    if next_page_token:
        params["nextPageToken"] = next_page_token
    response = requests.get(f"{JIRA_BASE_URL}/rest/api/3/search/jql", headers=jira_auth_headers(), params=params, timeout=60)
    result = safe_json_response(response)
    result["projectKey"] = project_key
    result["jql"] = jql
    return result


@app.get("/api/jira/issue/{issue_key}")
def jira_get_issue(issue_key: str):
    response = requests.get(f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}", headers=jira_auth_headers(), params={"fields": "summary,description,status,assignee,issuetype,labels,priority,components,created,updated,reporter,parent"}, timeout=30)
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
        return JSONResponse(status_code=400, content={"error": "issueKey is required"})
    try:
        issue = get_issue(issue_key)
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    normalized = normalize_live_issue(issue)
    return {"issueKey": issue_key, "normalized_issue": normalized, "assessment": evaluate_ticket(normalized)}


@app.post("/api/tickets/rewrite")
def rewrite_ticket(payload: Dict[str, Any]):
    return merge_human_input_into_rewrite((payload.get("summary") or "").strip(), (payload.get("description") or "").strip(), (payload.get("issueType") or payload.get("issue_type") or "Story").strip(), {})


@app.post("/api/tickets/rewrite-live")
def rewrite_live_ticket(payload: Dict[str, Any]):
    issue_key = (payload.get("issueKey") or "").strip().upper()
    if not issue_key:
        return JSONResponse(status_code=400, content={"error": "issueKey is required"})
    try:
        issue = get_issue(issue_key)
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    normalized = normalize_live_issue(issue)
    rewrite = merge_human_input_into_rewrite(normalized["summary"], normalized["description"], normalized["issue_type"] or "Story", {})
    return {"issueKey": issue_key, "normalized_issue": normalized, "rewrite": rewrite}


@app.post("/api/tickets/enhance-with-input")
def enhance_with_human_input(payload: Dict[str, Any]):
    issue_key = (payload.get("issueKey") or "").strip().upper()
    if not issue_key:
        return JSONResponse(status_code=400, content={"error": "issueKey is required"})
    try:
        issue = get_issue(issue_key)
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    normalized = normalize_live_issue(issue)
    enhanced = merge_human_input_into_rewrite(normalized["summary"], normalized["description"], normalized["issue_type"] or "Story", payload.get("human_input") or {})
    return {"issueKey": issue_key, "normalized_issue": normalized, "human_input": payload.get("human_input") or {}, "enhanced_rewrite": enhanced}


@app.post("/api/tickets/auto-clean")
def auto_clean_ticket(payload: Dict[str, Any]):
    return auto_clean_and_normalize_ticket((payload.get("summary") or "").strip(), (payload.get("description") or "").strip(), (payload.get("issueType") or payload.get("issue_type") or "Story").strip())


@app.post("/api/tickets/auto-clean-live")
def auto_clean_live_ticket(payload: Dict[str, Any]):
    issue_key = (payload.get("issueKey") or "").strip().upper()
    if not issue_key:
        return JSONResponse(status_code=400, content={"error": "issueKey is required"})
    try:
        issue = get_issue(issue_key)
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    normalized = normalize_live_issue(issue)
    cleaned = auto_clean_and_normalize_ticket(normalized["summary"], normalized["description"], normalized["issue_type"] or "Story")
    return {"issueKey": issue_key, "normalized_issue": normalized, "cleaned": cleaned}


@app.post("/api/tickets/bulk-dashboard")
def bulk_dashboard(payload: Dict[str, Any]):
    project_key = normalize_project_key(payload.get("projectKey"))
    jql = payload.get("jql") or f"project = {project_key} ORDER BY created DESC"
    max_results = int(payload.get("maxResults", 25))
    try:
        return build_bulk_dashboard(project_key, jql, max_results)
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/tickets/bulk-dashboard.csv")
def bulk_dashboard_csv(projectKey: str = Query(default=DEFAULT_PROJECT), jql: Optional[str] = Query(default=None), maxResults: int = Query(default=25)):
    project_key = normalize_project_key(projectKey)
    resolved_jql = jql or f"project = {project_key} ORDER BY created DESC"
    result = build_bulk_dashboard(project_key, resolved_jql, maxResults)
    output = csv_stream(bulk_results_to_csv_rows(result, weak_only=False))
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{project_key.lower()}_ticket_quality_dashboard.csv"'})


@app.get("/api/tickets/bulk-dashboard-weak.csv")
def bulk_dashboard_weak_csv(projectKey: str = Query(default=DEFAULT_PROJECT), jql: Optional[str] = Query(default=None), maxResults: int = Query(default=25)):
    project_key = normalize_project_key(projectKey)
    resolved_jql = jql or f"project = {project_key} ORDER BY created DESC"
    result = build_bulk_dashboard(project_key, resolved_jql, maxResults)
    output = csv_stream(bulk_results_to_csv_rows(result, weak_only=True))
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{project_key.lower()}_weak_tickets.csv"'})


@app.post("/api/tickets/suggest-breakdown")
def api_suggest_breakdown(payload: Dict[str, Any]):
    issue_key = (payload.get("issueKey") or "").strip().upper()
    mode = (payload.get("mode") or "standard").strip().lower()
    if not issue_key:
        return JSONResponse(status_code=400, content={"error": "issueKey is required"})
    try:
        issue = get_issue(issue_key)
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return suggest_breakdown(issue, mode=mode)


@app.post("/api/jira/create-batch")
def jira_create_batch(payload: Dict[str, Any]):
    project_key = normalize_project_key(payload.get("projectKey"))
    children = payload.get("children") or []
    if not isinstance(children, list) or not children:
        return JSONResponse(status_code=400, content={"error": "children array is required"})
    results = [create_single_issue(project_key, child) for child in children]
    return {"projectKey": project_key, "createdCount": sum(1 for r in results if r["status_code"] in {200, 201}), "results": results}


@app.post("/api/jira/create")
def jira_create(data: Dict[str, Any]):
    project_key = normalize_project_key(data.get("projectKey"))
    fields: Dict[str, Any] = {
        "project": {"key": project_key},
        "summary": data.get("summary", "").strip(),
        "description": build_adf_document(data.get("description", "")),
        "issuetype": {"name": data.get("issueType", "Story")},
    }
    if not fields["summary"]:
        return JSONResponse(status_code=400, content={"error": "summary is required"})
    if data.get("labels"):
        fields["labels"] = normalize_to_list(data.get("labels"))
    if data.get("components"):
        fields["components"] = [{"name": x} for x in normalize_to_list(data.get("components"))]
    if data.get("parentIssueKey"):
        fields["parent"] = {"key": data.get("parentIssueKey")}
    if str(data.get("issueType", "Story")).strip().lower() == "story" and data.get("storyPoints") is not None:
        fields[STORY_POINTS_FIELD_ID] = data.get("storyPoints")
    response = requests.post(f"{JIRA_BASE_URL}/rest/api/3/issue", headers=jira_auth_headers(), json={"fields": fields}, timeout=30)
    result = safe_json_response(response)
    result["projectKey"] = project_key
    return result


@app.post("/api/jira/update")
def jira_update(data: Dict[str, Any]):
    issue_key = (data.get("issueKey") or "").strip().upper()
    if not issue_key:
        return JSONResponse(status_code=400, content={"error": "issueKey is required"})
    fields_to_update: Dict[str, Any] = {}
    if data.get("summary") is not None and str(data.get("summary")).strip():
        fields_to_update["summary"] = str(data.get("summary")).strip()
    if data.get("description") is not None:
        fields_to_update["description"] = build_adf_document(str(data.get("description")))
    if data.get("issueType") is not None and str(data.get("issueType")).strip():
        fields_to_update["issuetype"] = {"name": str(data.get("issueType")).strip()}
    if data.get("labels") is not None:
        fields_to_update["labels"] = normalize_to_list(data.get("labels"))
    if data.get("components") is not None:
        fields_to_update["components"] = [{"name": x} for x in normalize_to_list(data.get("components"))]
    if str(data.get("issueType", "")).strip().lower() == "story" and data.get("storyPoints") is not None:
        fields_to_update[STORY_POINTS_FIELD_ID] = data.get("storyPoints")
    if not fields_to_update:
        return JSONResponse(status_code=400, content={"error": "At least one field must be provided"})
    response = requests.put(f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}", headers=jira_auth_headers(), json={"fields": fields_to_update}, timeout=30)
    result = {"status_code": response.status_code, "issueKey": issue_key}
    if response.text.strip():
        try:
            result["jira_response"] = response.json()
        except Exception:
            result["jira_response"] = {"raw_text": response.text}
    else:
        result["jira_response"] = {"message": "Issue updated successfully"}
    return result
