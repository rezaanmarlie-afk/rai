# Rovo-Integrated Compliance Platform

This prototype gives you a practical starting point for a Jira compliance platform that can sit behind an Atlassian Rovo Agent.

## What is included
- FastAPI backend
- Simple dashboard page
- Jira search and create endpoints
- Ticket drafting endpoint
- Compliance scoring engine
- Rovo webhook adapter for a Forge Agent
- Forge starter manifest and action example

## Why this design
There is no general-purpose direct Ask Rovo chat API to call like a normal LLM endpoint. The reliable Atlassian-supported options are:
1. Build a Forge Rovo Agent that invokes actions
2. Use the Atlassian Rovo MCP Server from supported MCP clients
3. Keep your deterministic business logic in your own service

This prototype follows option 3 plus a Forge handoff.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --reload --port 8000
```

Open `http://127.0.0.1:8000`

## Example API calls

### 1) Draft a better Jira story
```bash
curl -X POST http://127.0.0.1:8000/api/tickets/draft \
  -H "Content-Type: application/json" \
  -d '{
    "summary": "TMF641 integration for service order orchestration",
    "business_context": "Standardize service ordering between front-end workflow and orchestration layer.",
    "integration_type": "API",
    "api_standard": "TMF641",
    "data_source": "Oracle UIM",
    "dependencies": ["HCO available", "Architecture sign-off"]
  }'
```

### 2) Check ticket compliance
```bash
curl -X POST http://127.0.0.1:8000/api/compliance/check \
  -H "Content-Type: application/json" \
  -d '{
    "issues": [
      {
        "key": "OSS-101",
        "fields": {
          "summary": "TMF641 integration for service order orchestration",
          "description": "Build REST API for service orders using TMF641 against Oracle UIM. Include acceptance criteria and dependency on HCO availability.",
          "labels": ["api", "tmf641", "oracle uim"]
        }
      }
    ]
  }'
```

### 3) Rovo webhook shape
```bash
curl -X POST http://127.0.0.1:8000/api/rovo/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "action": "check_compliance",
    "payload": {
      "issues": [
        {
          "key": "OSS-101",
          "fields": {
            "summary": "TMF641 integration for service order orchestration",
            "description": "Build REST API for service orders using TMF641 against Oracle UIM. Include acceptance criteria and dependency on HCO availability.",
            "labels": ["api", "tmf641", "oracle uim"]
          }
        }
      ]
    }
  }'
```

## Jira custom fields
The sample app uses semantic field names like `integration_type`, `api_standard`, `data_source`, `acceptance_criteria`, and `dependencies` inside the compliance engine.

For real Jira integration you should map those to your actual custom field IDs, for example:
- `customfield_12345` = Integration Type
- `customfield_12346` = API Standard
- `customfield_12347` = Data Source

## Suggested next step for your environment
- Add your Jira custom field IDs
- Add project-specific JQL presets
- Add Vodacom/ASOC rule packs
- Wrap the `/api/rovo/webhook` endpoint with a Forge Agent action
