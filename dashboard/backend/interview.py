"""Multi-turn interview agent — gathers problem spec through Q&A, returns params.

The agent asks one question at a time. When it has all required parameters it
responds with a JSON completion block that the frontend uses to kick off optimization.
"""
import json
import anthropic

SYSTEM_PROMPT = """You are a structural optimization intake assistant inside an engineering design tool.

Your job: gather all required parameters for optimization through natural, brief conversation.
Ask ONE question at a time. Keep every response short — one or two sentences max.

SUPPORTED DOMAINS:

1. BEAM — cantilever beam
   Required: beam_length_mm, tip_load_N
   Optional (use defaults if not raised): H_init_mm=10, material="Al-6061",
     deflection_limit_mm=0.05, stress_limit_mpa=165.6

2. GEAR — spur gear pinion
   Required: torque_Nm
   Optional (use defaults if not raised): num_teeth=15, material="AISI 4140",
     m_init_mm=2.0, b_init_mm=25.0

RULES:
- Start by asking what they want to optimize (beam or gear).
- Ask only about REQUIRED parameters — do not ask about optional ones unless the user
  raises them first. Use defaults silently.
- When you have all required params, confirm them in one sentence, then respond ONLY
  with the JSON block below (no other text before or after it).

COMPLETION JSON — beam:
{"done":true,"domain":"beam","summary":"<one sentence description>","params":{"L":<meters>,"load":<N>,"H_init":0.010}}

COMPLETION JSON — gear:
{"done":true,"domain":"gear","summary":"<one sentence description>","params":{"torque":<Nm>,"m_init":0.002,"b_init":0.025}}

Example beam: {"done":true,"domain":"beam","summary":"Optimize a 150 mm Al-6061 cantilever beam under 800 N tip load.","params":{"L":0.150,"load":800.0,"H_init":0.010}}
"""


def send_message(messages: list, api_key: str) -> dict:
    """Send one turn of the interview.

    messages: list of {role, content} dicts (growing conversation history)
    Returns: {role, content, done, domain?, params?, summary?}
    """
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=messages,
    )

    text = response.content[0].text.strip()

    # Detect completion JSON — agent may prepend a confirmation sentence
    json_start = text.rfind('{"done":true')
    if json_start == -1:
        json_start = text.rfind('{"done": true')

    if json_start != -1:
        try:
            parsed = json.loads(text[json_start:])
            if parsed.get("done") is True:
                return {
                    "role":    "assistant",
                    "content": parsed.get("summary", "Parameters confirmed. Ready to optimize."),
                    "done":    True,
                    "domain":  parsed["domain"],
                    "params":  parsed["params"],
                    "summary": parsed.get("summary", ""),
                }
        except (json.JSONDecodeError, KeyError):
            pass

    return {"role": "assistant", "content": text, "done": False}
