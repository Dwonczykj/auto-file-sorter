from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import json
from datetime import datetime

from gmail_rule_daemon import GmailAutomation, GmailRuleEngine, AIService, EmailRule

app = FastAPI(title="Gmail Rule Daemon API")

# Initialize services
ai_service = AIService.get_instance(model_name="gpt-4")
gmail = GmailAutomation(
    credentials_path='/Users/joey/Github_Keep/python_scripts/auto-file-sorter/Google Cloud Credentials.json',
    token_path='/Users/joey/Github_Keep/python_scripts/auto-file-sorter/gmail_token.json',
    ai_service=ai_service
)
rule_engine = GmailRuleEngine(gmail, 'email_rules.json')

# Pydantic models for request/response


class RuleCondition(BaseModel):
    from_: Optional[str] = None
    to: Optional[str] = None
    subject: Optional[str] = None


class RuleAction(BaseModel):
    type: str
    value: Optional[str] = None


class Rule(BaseModel):
    name: str
    conditions: Dict[str, str]
    actions: List[Dict[str, Any]]


class RuleCreate(BaseModel):
    name: str
    conditions: RuleCondition
    actions: List[RuleAction]


class AIRulePrompt(BaseModel):
    prompt: str


class UnsubscribeRequest(BaseModel):
    folder_name: str
    max_emails: int = 100


class FilteredEmail(BaseModel):
    id: str
    from_: str
    subject: str
    date: datetime
    applied_rules: List[str]


@app.get("/rules", response_model=List[Rule])
async def get_rules():
    """Get all email rules"""
    try:
        with open('email_rules.json', 'r') as f:
            rules = json.load(f)
        return rules
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rules", response_model=Rule)
async def create_rule(rule: RuleCreate):
    """Create a new email rule manually"""
    try:
        new_rule = {
            "name": rule.name,
            "conditions": {
                k: v for k, v in rule.conditions.dict(by_alias=True).items()
                if v is not None
            },
            "actions": [action.dict() for action in rule.actions]
        }
        rule_engine._add_rule_to_file(new_rule)
        return new_rule
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rules/ai", response_model=Rule)
async def create_rule_from_ai(prompt: AIRulePrompt):
    """Create a new email rule using AI"""
    try:
        await rule_engine.create_rule_from_prompt(prompt.prompt)
        # Return the last created rule
        with open('email_rules.json', 'r') as f:
            rules = json.load(f)
        return rules[-1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/filtered-emails", response_model=List[FilteredEmail])
async def get_filtered_emails(limit: int = 100):
    """Get recently filtered emails"""
    try:
        messages = gmail.service.users().messages().list(
            userId='me',
            maxResults=limit,
            q='has:userlabels'
        ).execute()

        filtered_emails = []
        if 'messages' in messages:
            for msg in messages['messages']:
                full_msg = gmail.service.users().messages().get(
                    userId='me', id=msg['id'], format='metadata'
                ).execute()

                headers = {h['name']: h['value']
                           for h in full_msg['payload']['headers']}
                labels = full_msg.get('labelIds', [])

                filtered_emails.append(FilteredEmail(
                    id=msg['id'],
                    from_=headers.get('From', ''),
                    subject=headers.get('Subject', ''),
                    date=datetime.fromtimestamp(
                        int(full_msg['internalDate'])/1000),
                    applied_rules=labels
                ))

        return filtered_emails
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unsubscribe")
async def process_unsubscribes(request: UnsubscribeRequest, background_tasks: BackgroundTasks):
    """Process unsubscribe requests for a folder"""
    try:
        # Run in background as it might take time
        background_tasks.add_task(
            rule_engine.gmail.process_unsubscribes,
            request.folder_name,
            request.max_emails
        )
        return {"message": f"Processing unsubscribe requests from folder: {request.folder_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs")
async def get_logs(limit: int = 100):
    """Get recent log entries"""
    try:
        log_file = Path("gmail_daemon.log")
        if not log_file.exists():
            return {"logs": []}

        with open(log_file, 'r') as f:
            logs = f.readlines()[-limit:]

        return {"logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/folders")
async def get_folders():
    """Get all Gmail folders/labels"""
    try:
        folders = await rule_engine.gmail.list_folders()
        return {"folders": folders}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required to run the API server. Install it with: pip install uvicorn"
        )

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logging.error(f"Failed to start API server: {e}")
