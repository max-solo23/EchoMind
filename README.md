---
title: EchoMind
app_file: app/app.py
sdk: gradio
sdk_version: 5.49.1
---
## Setup Instructions

```bash
# clone repository
git clone <repository-url>
cd <project-folder>

# create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# start the application
cd app
python app.py
```

## Environment Variables

Create your own .env file in the root directory of the project. This file is required to store your personal API keys and tokens securely. Do not commit it to version control.

Pushover credentials are optional and only needed if you want to enable notification features.

Example `.env` content:

```
PUSHOVER_USER=youruser
PUSHOVER_TOKEN=yourtoken
OPENAI_API_KEY=yourkey
# or set AZURE_OPENAI_API_KEY instead of OPENAI_API_KEY
OPENAI_MODEL=gpt-5-nano
```
Replace the placeholder values with your actual credentials before running the app.

## Profile Configuration

Generate a persona profile by creating `app/persona.yaml` manually. The application will not create this file for you, and the repository may ship without one so that you can supply your own details. Once the file exists, it defines metadata about the persona, the skill set, project history, and preferred response style used by the application. Make sure to keep this file inside the `app` directory so it can be loaded when you launch the UI from the same location.

Skeleton structure:

```
name:
title:
summary: >
  [Brief professional overview. Describe focus areas, motivations, and general goals.]

skills:
  programming_languages: []
  frameworks_tools: []

projects:
  - name:
    role:
    tech_stack: []
    description: >
      [Short description of what the project does or demonstrates.]
  - name:
    role:
    tech_stack: []
    description: >
      [Short description of what the project does or demonstrates.]

education:
  - name:
    courses: []

contacts:
  github: ""
  linkedin: ""
  huggingface: ""

style:
  tone: factual
  response_length: short
  language_priority: English
```
