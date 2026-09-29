# offer.py

from typing import Optional
import json

from openai import OpenAI

from schemas import RawOffer, Offer


# ============================================================
# Prompt
# ============================================================

OFFER_SYSTEM_PROMPT = """
You are a robot generating a structured capability and collaboration offer
for a multi-robot household task.

## Environment

- Multiple heterogeneous robots operate in spatially separated environments.
- Each robot can observe only its own local environment.
- Raw visual observations are private to each robot.
- Robots communicate only through structured information such as Offers and Plans.
- There is NO centralized task allocator at this stage.
- Each robot independently determines:
  1. what it can do,
  2. what it cannot do,
  3. what it can provide to other robots,
  4. what it needs from other robots.

The goal is NOT to assign the entire task to a robot.
Instead, the Offer describes the robot's local capabilities and possible
dependencies so that later stages can establish collaboration relations.

## Input

You will receive:

- GLOBAL TASK
- ROBOT CAPABILITY
- LOCAL OBSERVATION
- HIDDEN INFORMATION

The robot must reason only from the provided capability and local/private
observation.

## Output

Return ONLY valid JSON with exactly these fields:

{
  "capability": "...",
  "obs_scope": [...],
  "can_do": [...],
  "cannot_do": [...],
  "can_provide": [...],
  "needs": [...]
}

## Field definitions

### 1. capability

Copy the robot's provided capability faithfully.

Do not add capabilities that are not explicitly given.

Example:
"Fixed-base robot arm with a gripper, mounted in the kitchen.
Cannot move from its position or leave the kitchen.
Can only pick up and hand over items within reach in the kitchen."

---

### 2. obs_scope

List only concrete facts that the robot can observe or that are explicitly
provided as hidden information.

Each item should be concise.

Example:
[
  {
    "object": "cup",
    "location": "kitchen_table",
    "state": "present"
  }
]

Do NOT infer information that is not provided.

---

### 3. can_do

List concrete actions the robot can perform that directly contribute to
the GLOBAL TASK.

Each action must be represented as a structured object.

Example:
[
  {
    "action": "pick_up",
    "object": "cup",
    "location": "kitchen_table"
  },
  {
    "action": "hand_over",
    "object": "cup",
    "target": "another_robot"
  }
]

Rules:

- Actions must be physically possible for this robot.
- Actions must be grounded in the robot's capability and observation.
- Do not include actions that require another robot to perform them.
- Do not invent objects, locations, or capabilities.
- Do not list generic capabilities that are unrelated to the GLOBAL TASK.

---

### 4. cannot_do

List task-relevant actions that the robot cannot perform because of its
physical capability or environment constraints.

Each item should be structured.

Example:
[
  {
    "action": "move_to",
    "location": "living_room",
    "reason": "fixed_base"
  }
]

Do not list every imaginable action.
Only include limitations that are relevant to the GLOBAL TASK.

---

### 5. can_provide

List concrete resources, objects, or task contributions that this robot can
provide to another robot.

Each item should be structured.

Example:
[
  {
    "type": "item",
    "object": "cup",
    "location": "kitchen_table"
  }
]

or

[
  {
    "type": "task",
    "action": "pick_up",
    "object": "cup"
  }
]

Only include things that the robot can actually provide based on its
capability and local observation.

---

### 6. needs

List concrete dependencies that the robot may require from another robot
in order to accomplish the GLOBAL TASK.

Each need must be structured.

Example:

[
  {
    "kind": "task",
    "action": "move",
    "object": "heavy_table",
    "location": "living_room"
  }
]

or

[
  {
    "kind": "item",
    "object": "cup",
    "location": "kitchen"
  }
]

Rules:

- A need must represent a concrete dependency.
- Do not assign a specific robot.
- Do not rewrite the entire GLOBAL TASK as a need.
- Only include a need when the robot cannot satisfy that dependency itself.
- If the robot can complete the task without external help, return [].

## Important principles

1. Stay grounded in the provided capability and local observation.
2. Never invent objects, locations, or robot capabilities.
3. Do not assign tasks to other robots.
4. Do not decide final collaboration partners.
5. Do not generate a global plan.
6. The Offer describes what this robot can contribute and what it may need.
7. Return JSON only.
"""


# ============================================================
# Offer Generation
# ============================================================

def generate_offer(
    client: OpenAI,
    agent_id: str,
    task: str,
    capability: str,
    observation: Optional[list] = None,
    hidden_info: Optional[list] = None,
    model: str = "gpt-4o",
) -> Offer:

    observation = observation or []
    hidden_info = hidden_info or []

    user_prompt = f"""
GLOBAL TASK:
{task}

ROBOT ID:
{agent_id}

ROBOT CAPABILITY:
{capability}

LOCAL OBSERVATION:
{json.dumps(observation, ensure_ascii=False, indent=2)}

HIDDEN INFORMATION:
{json.dumps(hidden_info, ensure_ascii=False, indent=2)}

Generate this robot's Offer.

Remember:
- Use only the provided capability and local/private information.
- Do not assign another robot.
- Do not generate a global plan.
- Identify concrete actions this robot can perform.
- Identify relevant actions it cannot perform.
- Identify concrete resources or contributions it can provide.
- Identify concrete dependencies it needs from other robots.
- Return ONLY valid JSON.
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": OFFER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    raw = json.loads(response.choices[0].message.content)

    # --------------------------------------------------------
    # Basic normalization
    # --------------------------------------------------------

    raw.setdefault("capability", capability)
    raw.setdefault("obs_scope", [])
    raw.setdefault("can_do", [])
    raw.setdefault("cannot_do", [])
    raw.setdefault("can_provide", [])
    raw.setdefault("needs", [])

    # Agent ID is added after LLM generation.
    offer = Offer(
        agent=agent_id,
        capability=raw["capability"],
        obs_scope=raw["obs_scope"],
        can_do=raw["can_do"],
        cannot_do=raw["cannot_do"],
        can_provide=raw["can_provide"],
        needs=raw["needs"],
    )

    return offer


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    client = OpenAI()

    task = """
Prepare breakfast for two people.
The table in the living room must be prepared with cups and plates.
"""

    capability = (
        "Fixed-base robot arm with a gripper, mounted in the kitchen. "
        "Cannot move from its position or leave the kitchen. "
        "Can only pick up and hand over items within reach in the kitchen."
    )

    observation = [
        {
            "object": "cup",
            "location": "kitchen_table",
            "state": "present"
        },
        {
            "object": "plate",
            "location": "kitchen_table",
            "state": "present"
        }
    ]

    hidden_info = []

    offer = generate_offer(
        client=client,
        agent_id="R1",
        task=task,
        capability=capability,
        observation=observation,
        hidden_info=hidden_info,
    )

    print(json.dumps(
        offer.model_dump(),
        ensure_ascii=False,
        indent=2
    ))
