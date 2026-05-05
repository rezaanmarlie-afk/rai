from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import requests
import os
import base64

load_dotenv()

app = FastAPI()

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")

def jira_auth():
    token = f"{JIRA_EMAIL}:{JIRA_API_TOKEN}"
    return {
        "Authorization": "Basic " + base64.b64encode(token.encode()).decode(),
        "Content-Type": "application/json"
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>Jira Ticket Quality Tool</h1>
    <p>Use /docs to test APIs</p>
    """

# 🔹 Check ticket quality
@app.post("/api/tickets/check")
def check_ticket(data: dict):
    description = data.get("description", "")
    summary = data.get("summary", "")

    score = 100
    issues = []

    if len(summary) < 10:
        score -= 20
        issues.append("Summary too short")

    if len(description) < 30:
        score -= 30
        issues.append("Description too short")

    if "Given" not in description:
        score -= 20
        issues.append("No acceptance criteria (Given/When/Then)")

    if "dependency" not in description.lower():
        score -= 10
        issues.append("No dependencies mentioned")

    return {
        "score": score,
        "issues": issues,
        "status": "PASS" if score >= 70 else "FAIL"
    }

# 🔹 Jira search
@app.post("/api/jira/search")
def jira_search(body: dict):
    jql = body.get("jql", f"project = {JIRA_PROJECT_KEY} ORDER BY created DESC")
    max_results = body.get("maxResults", 10)
    next_page_token = body.get("nextPageToken")
    fields = body.get("fields", ["summary", "status", "assignee"])

    url = f"{JIRA_BASE_URL}/rest/api/3/search/jql"
    params = {
        "jql": jql,
        "maxResults": max_results,
        "fields": ",".join(fields),
        "fieldsByKeys": "false",
    }
    if next_page_token:
        params["nextPageToken"] = next_page_token

    response = requests.get(url, headers=jira_auth(), params=params)

    try:
        jira_json = response.json()
    except Exception:
        jira_json = {"raw_text": response.text}

    return {
        "status_code": response.status_code,
        "jira_response": jira_json
    }

@app.get("/api/jira/issue/{issue_key}")
def jira_get_issue(issue_key: str):
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}"
    params = {
        "fields": "summary,description,status,assignee,issuetype,labels,priority,components,created,updated,reporter"
    }
    response = requests.get(url, headers=jira_auth(), params=params)

    try:
        jira_json = response.json()
    except Exception:
        jira_json = {"raw_text": response.text}

    return {
        "status_code": response.status_code,
        "jira_response": jira_json
    }