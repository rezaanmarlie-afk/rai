import api, { fetch } from '@forge/api';

const BASE_URL = 'https://your-compliance-platform.example.com';

export async function compliance(payload) {
  const issues = payload.issues ? JSON.parse(payload.issues) : [];
  const response = await fetch(`${BASE_URL}/api/rovo/webhook`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      action: 'check_compliance',
      payload: { issues }
    })
  });
  const data = await response.json();
  return {
    success: true,
    output: JSON.stringify(data)
  };
}

export async function draft(payload) {
  const response = await fetch(`${BASE_URL}/api/rovo/webhook`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      action: 'draft_ticket',
      payload: {
        summary: payload.summary,
        business_context: payload.business_context
      }
    })
  });
  const data = await response.json();
  return {
    success: true,
    output: JSON.stringify(data)
  };
}
