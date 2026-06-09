# Section 4: Prompting and APIs — OpenAI API & Tool Calling
**Lecture 4 | IIT Kharagpur × upGrad | Instructor: Prof. Sourangshu Bhattacharya**

> **Learning Goal:** Write production-ready code that talks to an LLM API. Understand the role system, conversation history, structured outputs, and tool/function calling — the four primitives that turn a chat endpoint into an application platform.

---

## Table of Contents
1. [OpenAI API — Introduction & Overview](#intro)
2. [OpenAI Model Families](#models)
3. [Basic Chat Completion — The 4-Step Flow](#basic-chat)
4. [Roles in Prompting](#roles)
5. [What Happens Under the Hood](#under-the-hood)
6. [Elements of a Prompt (API Context)](#elements-api)
7. [Tool / Function Calling](#tool-calling)
8. [Tool Calling — Full Worked Example: LCM Calculator](#lcm-example)
9. [Structured Outputs & JSON Mode](#json-mode)
10. [Practical Examples](#practical-examples)
11. [Anthropic API (Claude)](#anthropic)
12. [API vs Model Families Summary](#summary-table)
13. [Interview Questions](#interview)
14. [Key Learning Thoughts](#learning-thoughts)

---

## 1. OpenAI API — Introduction & Overview {#intro}

### What Is the OpenAI API?
A **proprietary, pay-per-token REST API** that gives you programmatic access to OpenAI's language models. It is the industry-standard LLM platform.

### Core Capabilities

| Capability | What it does |
|---|---|
| **Chat completions** | Primary LLM interaction — send messages, get responses |
| **Function / Tool calling** | Model calls external functions via structured outputs |
| **Structured outputs (JSON mode)** | Guarantees valid JSON schema-compliant responses |
| **Embeddings & retrieval** | Convert text to vectors for semantic search |

### Modalities Supported

| Modality | Examples |
|---|---|
| Text generation | Chat, summarization, classification, code |
| Code generation | Python, JS, SQL, and more |
| Image generation & vision | Generate images, analyze images |
| Audio | Speech-to-text (Whisper), text-to-speech |

### Key Advantages
- **High reliability:** Production-ready infrastructure, SLAs
- **Scalable:** Handles millions of requests per day
- **Wide model ecosystem:** Multiple models for different speed/cost/capability tradeoffs
- **Pay-per-token:** No upfront cost, scales with usage

### Pricing Model
You pay for:
- **Input tokens** — the tokens in your prompt/messages
- **Output tokens** — the tokens the model generates
- Different models have different prices per million tokens

---

## 2. OpenAI Model Families {#models}

### Frontier Models (General Intelligence)

#### GPT-5.4
- Most advanced model for **intelligence at scale**
- Strong in **agentic workflows and complex reasoning**
- High accuracy for professional and enterprise use cases
- Handles long, multi-step tasks efficiently

#### GPT-5.4 mini
- Cost-efficient version of GPT-5.4
- Optimized for **high-volume and low-latency** tasks
- Good balance between performance and cost
- Suitable for scalable applications where GPT-5.4 would be overkill

**Pricing snapshot (approximate):**
| Model | Input | Output |
|---|---|---|
| GPT-5.4 mini | $0.75 / 1M tokens | $4.50 / 1M tokens |
| GPT-4o-mini | ~$0.15 / 1M tokens | ~$0.60 / 1M tokens |

### Coding & Reasoning Models

#### GPT-5.3-Codex
- Highly capable **agentic coding model**
- Handles **large codebases and multi-step** development tasks
- Strong debugging and refactoring capabilities
- Suitable for advanced developer tools (think: AI IDE, code review bots)

#### GPT-5.1-Codex
- Optimized for **structured programming tasks**
- More efficient for **mid-level coding workflows**
- Good balance between speed and capability
- Useful for automation and backend development

### Image Generation Models

#### GPT Image 1.5
- State-of-the-art **image generation quality**
- Produces highly detailed and realistic visuals
- Supports advanced prompt control
- For professional design and media workflows

#### GPT Image 1
- Earlier generation — faster and more cost-efficient
- Good for basic image generation tasks
- Suitable for lightweight applications

### How to Choose a Model
```
Is it a chat/reasoning task?
  → Complex, multi-step, agentic?  → GPT-5.4
  → High volume, cost-sensitive?  → GPT-5.4 mini or GPT-4o-mini

Is it a coding/debugging task?
  → Large codebase, complex refactor? → GPT-5.3-Codex
  → Standard programming task?      → GPT-5.1-Codex

Is it image generation?
  → Professional quality needed? → GPT Image 1.5
  → Basic/lightweight?          → GPT Image 1
```

---

## 3. Basic Chat Completion — The 4-Step Flow {#basic-chat}

### The Flow
```
Step 1: User sends messages to API
Step 2: Model processes system + user input
Step 3: Model generates response
Step 4: API returns structured output
```

### Minimal Working Example

```python
from openai import OpenAI

# Reads OPENAI_API_KEY from environment variable automatically
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Explain RAG in one sentence."}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
```

### The Response Object Structure
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "RAG (Retrieval-Augmented Generation) is a technique..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 47,
    "total_tokens": 75
  }
}
```

**Key fields:**
- `choices[0].message.content` — the generated text
- `choices[0].finish_reason` — why generation stopped (`"stop"`, `"length"`, `"tool_calls"`)
- `usage.total_tokens` — how many tokens were consumed (affects billing)

---

## 4. Roles in Prompting {#roles}

The OpenAI API structures conversations as a **messages array**, where each message has a `role` and `content`. This is the most important concept to understand for building API applications.

### The Four Roles

#### 1. `system`
**Purpose:** Sets the overall behavior, tone, and constraints for the assistant.

```python
{"role": "system", "content": "You are a helpful assistant that only answers questions about Python programming. Refuse off-topic questions politely."}
```

- Read first, persists as the model's "instructions" for the entire conversation
- Controls: persona, tone, constraints, format requirements, topic focus
- Example uses: "You are a medical assistant. Always recommend consulting a doctor.", "Respond only in formal English."

#### 2. `developer` (New Role)
**Purpose:** Designed for developers to provide specialized instructions with more control than the system role.

```python
{"role": "developer", "content": "You translate text into emojis."}
```

- Newer role added for API-level developer control
- Gives more granular control over API interaction than the traditional system role
- Useful for separating "system behavior" from "developer-level technical instructions"

#### 3. `user`
**Purpose:** Represents the input from the human user — questions, instructions, or any user-generated content.

```python
{"role": "user", "content": "What is the capital of France?"}
```

- This is the actual human input in each turn
- In production apps, this is filled dynamically from user input

#### 4. `assistant`
**Purpose:** Represents the model's previous responses — used to maintain conversation history and to inject few-shot examples.

```python
{"role": "assistant", "content": "Paris is the capital of France."}
```

**Two uses of the assistant role:**
1. **Conversation history:** Append previous model responses to maintain context across turns
2. **Few-shot prompting:** Inject "example" assistant responses to steer output style/format

### Why Roles Matter — The Main Purpose

> **From class Q&A:** The main purpose of roles is to **provide structure to the prompt engineering process**. They allow: (1) system-level instructions that persist; (2) structured inputs to be fed to the model; (3) the model to understand who is speaking (user vs. assistant) to maintain conversational coherence.

---

## 5. What Happens Under the Hood {#under-the-hood}

The structured messages array is **NOT sent to the model as JSON**. Before reaching the model, it is flattened into a **single token sequence**:

```
Structured messages:          What the model actually sees:
┌─────────────────────┐       ┌─────────────────────────────────┐
│ role: "system"      │       │ <|system|>                      │
│ content: "You are   │  →    │ You are helpful                 │
│          helpful"   │       │ <|user|>                        │
│                     │       │ Hello                           │
│ role: "user"        │       │ <|assistant|>                   │
│ content: "Hello"    │       │ Hi!                             │
│                     │       └─────────────────────────────────┘
│ role: "assistant"   │
│ content: "Hi!"      │
└─────────────────────┘
```

The role markers become **special tokens** (like `<|system|>`, `<|user|>`, `<|assistant|>`) that the model was trained to recognize. To the Transformer, this is just a sequence of tokens — the structure is encoded in the special tokens.

**Why this matters:** You can't inject arbitrary text into the "system" position at inference time without going through the API's role system. The structure is intentional and enforced.

---

## 6. Elements of a Prompt (API Context) {#elements-api}

The four prompt elements (from Section 2) map directly onto the API's role system:

| Element | API Implementation |
|---|---|
| **Instructions** | `system` message or start of `user` message |
| **Context** | `system` message or prepended to `user` message |
| **Input Data** | `user` message content |
| **Output Indicator** | End of `user` message or `assistant` message prefix |

```python
messages = [
    # Instructions + Context → system role
    {
        "role": "system",
        "content": "Classify text into neutral, negative or positive. You are analyzing product reviews for an e-commerce platform."
    },
    # Input Data → user role
    {
        "role": "user",
        "content": "Text: I think the food was okay.\nSentiment:"
        # ↑ "Sentiment:" is the output indicator
    }
]
```

---

## 7. Tool / Function Calling {#tool-calling}

### What Is Tool Calling?
**Tool calling** (also called function calling) lets the model call external functions via structured outputs.

The model **converts natural language → function arguments** — it understands what function to call and what parameters to pass, then your code executes the function.

```
User query → Model → Tool selection → Function call → Response
                              ↓
                    Your Python code executes the function
                              ↓
                    Result sent back to model
                              ↓
                    Model produces final answer
```

### Key Features
- **JSON schema-based inputs:** Functions are described using JSON schemas; model generates valid arguments
- **Automatic tool selection:** Model decides which tool to call based on the user query
- **Reliable structured outputs:** Arguments are always valid JSON matching the schema
- **Integrates:** APIs, databases, services, compute functions

### Why Use Tool Calling Instead of Just Asking the LLM?

**Without tool calling — asking LLM to calculate LCM:**
```
"What is the LCM of 84 and 120?"
→ Model explains prime factorization step by step
→ Total tokens: ~425
→ Risk: model might make arithmetic errors
```

**With tool calling:**
```
→ Model calls compute_lcm(84, 120)
→ Python executes: returns 840 instantly
→ Total tokens: ~123
→ Result: guaranteed mathematically correct
```

**Tool calling wins on:**
1. **Accuracy** — code doesn't hallucinate; 2+2=4 every time
2. **Token efficiency** — 123 tokens vs 425 tokens (71% cheaper)
3. **Determinism** — same inputs always produce same outputs
4. **Real-world data access** — model can look up current weather, query a database, call an API

---

## 8. Tool Calling — Full Worked Example: LCM Calculator {#lcm-example}

### Step 1: Define the Tool Schema

```python
from openai import OpenAI
import json
import math

client = OpenAI()

# Define the function schema for the model
tools = [
    {
        "type": "function",
        "function": {
            "name": "compute_lcm",
            "description": "Computes the Least Common Multiple (LCM) of two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "The first integer."},
                    "b": {"type": "integer", "description": "The second integer."}
                },
                "required": ["a", "b"],
                "additionalProperties": False
            }
        }
    }
]

# The actual Python implementation
def compute_lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)
```

### Step 2: First API Call — Model Selects the Tool

```python
input_list = [
    {"role": "user", "content": "What is the LCM of 84 and 120?"}
]

# First LLM call: model reads the query, selects the tool, generates arguments
response = client.responses.create(
    model="gpt-4o-mini",
    tools=tools,
    input=input_list,
)

# Save the model's response (including tool call) to input_list
input_list += response.output
```

### Step 3: Execute the Function, Feed Result Back

```python
for item in response.output:
    if item.type == "function_call":
        if item.name == "compute_lcm":
            # Parse the arguments the model generated
            args = json.loads(item.arguments)
            # Execute the actual Python function
            lcm = compute_lcm(args["a"], args["b"])
            
            # Append the function result to the conversation
            input_list.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": f"LCM: {lcm}"
            })
```

### Step 4: Second API Call — Model Produces Final Answer

```python
# Second LLM call: model uses the function output to answer naturally
final_response = client.responses.create(
    model="gpt-4o-mini",
    instructions="Respond only with the LCM of the two integers provided.",
    tools=tools,
    input=input_list,
)

print(final_response.output_text)
# Output: "840"
# Total tokens: 123 (vs 425 without tool calling)
```

### The Full Flow Visualized

```
User: "What is the LCM of 84 and 120?"
         ↓
[LLM Call 1]
Model sees: user query + tool definitions
Model outputs: function_call(compute_lcm, {a: 84, b: 120})
         ↓
[Your Python Code]
compute_lcm(84, 120) → 840
         ↓
[LLM Call 2]  
Model sees: original query + function call + result (840)
Model outputs: "The LCM of 84 and 120 is 840."
```

### A More Complex Example: Weather Tool

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

# Usage with chat completions API:
response = client.chat.completions.create(
    model="gpt-4o",
    messages=msgs,
    tools=tools,
    tool_choice="auto"   # model decides whether to call a tool
)
```

`tool_choice` options:
- `"auto"` — model decides whether to call a tool
- `"required"` — model must call a tool
- `{"type": "function", "function": {"name": "get_weather"}}` — force a specific tool

---

## 9. Structured Outputs & JSON Mode {#json-mode}

### JSON Mode
Force the model to always return valid JSON:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extract entities from the text. Return JSON with keys: 'names', 'locations', 'dates'."},
        {"role": "user", "content": "Obama visited Paris in 2023."}
    ],
    response_format={"type": "json_object"}  # JSON mode
)
# Always returns valid JSON (but keys may vary)
```

### Structured Outputs (strict:true)
Guarantees schema compliance — model outputs ONLY the exact structure you specify:

```python
from pydantic import BaseModel

class Entity(BaseModel):
    names: list[str]
    locations: list[str]
    dates: list[str]

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[...],
    response_format=Entity,   # Pydantic schema
)
entity = response.choices[0].message.parsed
# entity.names = ["Obama"]
# entity.locations = ["Paris"]
# entity.dates = ["2023"]
```

### JSON Mode vs Structured Outputs

| Feature | JSON Mode | Structured Outputs |
|---|---|---|
| Always valid JSON? | Yes | Yes |
| Schema compliance? | Not guaranteed | Guaranteed |
| How to set | `response_format={"type": "json_object"}` | `response_format=YourPydanticModel` |
| Best for | Flexible JSON extraction | Exact schema enforcement |

---

## 10. Practical Examples {#practical-examples}

### Example 1: Chatbot Application

```python
from openai import OpenAI
client = OpenAI()

def chatbot(user_message: str, history: list) -> tuple[str, list]:
    """Multi-turn chatbot with conversation history."""
    history.append({"role": "user", "content": user_message})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful chatbot."},
            *history   # include full conversation history
        ]
    )
    
    assistant_reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_reply})
    
    return assistant_reply, history

# Usage
history = []
reply, history = chatbot("What is AI?", history)
print(reply)  # "AI is the simulation of human intelligence by machines."
reply, history = chatbot("Give me an example.", history)
print(reply)  # Model remembers previous context, gives AI example
```

**Key points:**
- **Multi-turn conversation:** Include full history in every API call
- **Context-aware responses:** Model sees all previous turns
- **Real-time interaction:** Each user message appended, response appended

### Example 2: Image Generation

```python
response = client.images.generate(
    model="gpt-image-1.5",
    prompt="A futuristic city at sunset",
    n=1,              # number of images
    size="1024x1024"
)

image_url = response.data[0].url
```

**Key points:**
- Different API endpoint (`images.generate` vs `chat.completions.create`)
- Model specified as image model
- Output is an image URL or base64

### Example 3: Conversation History with Few-Shot via Assistant Role

```python
messages = [
    {"role": "developer", "content": "You translate text into emojis."},
    # Few-shot examples via assistant role
    {"role": "user",      "content": "I love coffee."},
    {"role": "assistant", "content": "❤️ ☕"},
    {"role": "user",      "content": "The weather is sunny."},
    {"role": "assistant", "content": "☀️ 🌤️"},
    # Actual query
    {"role": "user",      "content": "I am going for a run."}
]
```

The model sees 2 input-output examples before the real query — classic few-shot prompting via the message roles. No separate "examples" parameter needed.

---

## 11. Anthropic API (Claude) {#anthropic}

The Anthropic Claude API is the main alternative to OpenAI. The lecture shows it alongside OpenAI.

### Key Differences from OpenAI API

| Aspect | OpenAI | Anthropic (Claude) |
|---|---|---|
| Client init | `OpenAI()` | `anthropic.Anthropic()` |
| Message structure | `chat.completions.create(messages=[...])` | `client.messages.create(messages=[...])` |
| Output access | `response.choices[0].message.content` | `response.content[0].text` |
| System message | Inside `messages` array with `"role": "system"` | Separate `system=` parameter |
| Max tokens | `max_tokens=` | `max_tokens=` (required, not optional) |

### Anthropic API Example

```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

response = client.messages.create(
    model="claude-sonnet-4-6",    # Current Sonnet model
    max_tokens=512,               # Required parameter
    system="You are a helpful assistant.",  # System message is separate
    messages=[
        {"role": "user", "content": "Write a poem about AI"}
    ],
    # Anthropic-style params (equivalent to OpenAI's):
    # temperature=1.1  (creative)
    # top_p=0.9        (nucleus)
    # top_k=50         (top-k)
    # stop=['\n\n']    (stop sequences)
)

print(response.content[0].text)
```

### Use-Case Presets (from lecture)

| Use Case | Config |
|---|---|
| **Factual Q&A** | `temp=0.3, top_p=0.7, max_tokens=256` |
| **Creative Writing** | `temp=1.2, top_p=0.95, rep_penalty=1.2` |
| **Code Generation** | `temp=0.2, top_k=20, stop=["\n\`\`\`"]` |
| **Brainstorming** | `temp=1.5, top_k=80, max_tokens=1024` |

---

## 12. API vs Model Families Summary {#summary-table}

| Model Family | Best For | API | Speed | Cost |
|---|---|---|---|---|
| GPT-5.4 | Complex reasoning, agentic | OpenAI | Slower | Higher |
| GPT-5.4 mini | High-volume apps | OpenAI | Fast | Lower |
| GPT-4o / 4o-mini | General chat, balanced | OpenAI | Fast | Medium |
| GPT-5.3-Codex | Complex coding, large repos | OpenAI | Slower | Higher |
| GPT-5.1-Codex | Standard coding | OpenAI | Fast | Medium |
| GPT Image 1.5 | Professional image gen | OpenAI | Medium | Higher |
| Claude Opus 4.8 | Complex reasoning | Anthropic | Slower | Highest |
| Claude Sonnet 4.6 | Balanced general use | Anthropic | Medium | Medium |
| Claude Haiku 4.5 | Fast, lightweight tasks | Anthropic | Fastest | Lowest |

---

## 13. Interview Questions {#interview}

**Q1: What is the purpose of each role (system, user, assistant) in OpenAI API chat completions?**
> **Answer:** `system`: Sets the overall behavior, persona, tone, and constraints for the model — it's like the "director's instructions" that shape every response in the conversation. `user`: Represents the actual human input — what the person is asking or saying in each turn. `assistant`: Represents the model's previous responses — used to maintain conversation context across multiple turns and also to inject few-shot examples. Under the hood, all three are flattened into a single token sequence with special role tokens (`<|system|>`, `<|user|>`, `<|assistant|>`).

**Q2: What is tool calling and why is it preferred for mathematical computations over direct LLM answers?**
> **Answer:** Tool calling lets the model invoke external functions by generating structured JSON arguments matching a defined schema. For mathematical computations, tool calling is preferred because: (1) **Accuracy** — code executes exactly (no hallucination); `math.gcd(84, 120)` always returns 12. (2) **Efficiency** — using tool calling for LCM calculation used 123 tokens vs 425 tokens for the step-by-step LLM approach (71% cheaper). (3) **Determinism** — same inputs always give same result, unlike LLM generation. (4) **Speed** — Python executes in microseconds; generating a multi-step explanation takes seconds.

**Q3: What is the difference between JSON mode and Structured Outputs in OpenAI API?**
> **Answer:** Both guarantee valid JSON output. JSON mode (`response_format={"type": "json_object"}`) ensures the response is always parseable JSON, but the schema/keys may vary. Structured Outputs (`response_format=YourPydanticModel, strict=True`) guarantees the response matches EXACTLY the Pydantic schema you provide — specific keys, types, and structure. Use JSON mode for flexible extraction; use Structured Outputs when your code depends on specific fields being present with specific types.

**Q4: Explain the 4-step flow of tool calling in OpenAI API.**
> **Answer:** (1) **Define tools:** Describe callable functions with JSON schemas (name, description, parameter types). (2) **First LLM call:** Send user query + tool definitions. The model decides which tool to call and generates valid JSON arguments. (3) **Execute function:** Your Python code executes the actual function using the generated arguments. Append the result to the conversation. (4) **Second LLM call:** Send original conversation + function output. Model uses the real result to generate a natural-language final answer.

**Q5: How does the assistant role support few-shot prompting in the API?**
> **Answer:** By injecting `{"role": "assistant", "content": "example response"}` messages BEFORE the actual user query, you show the model the exact output style, format, and tone you want. For example: user: "I love coffee.", assistant: "❤️ ☕", user: "The weather is sunny.", assistant: "☀️ 🌤️", user: "I am going for a run." — the model infers it should respond with emojis matching the sentiment, without any explicit instruction needed. This is more powerful than describing the output format in instructions, because the model directly mimics the shown examples.

**Q6: What happens "under the hood" when you send a messages array to the OpenAI API?**
> **Answer:** The structured messages array (with role/content dicts) is flattened into a single token sequence before reaching the Transformer. Each role becomes a special token: `<|system|>`, `<|user|>`, `<|assistant|>`. The model processes this as one continuous sequence, not as structured JSON. This is why the model can "know" who is speaking — it's trained on data formatted with these special role tokens. The role structure is an API convenience that maps to the model's token-level training format.

**Q7: When should you use GPT-5.4 vs GPT-5.4 mini?**
> **Answer:** GPT-5.4 is for complex, multi-step, reasoning-heavy tasks — agentic workflows, intricate analysis, professional/enterprise use cases where quality matters most. GPT-5.4 mini is for high-volume, cost-sensitive, latency-sensitive applications where the task is more straightforward — chatbots, simple Q&A, classification, summarization at scale. The mini variant offers good performance at significantly lower cost per token.

**Q8: What is the `finish_reason` field in the API response and why does it matter?**
> **Answer:** `finish_reason` tells you WHY the model stopped generating. Values: `"stop"` — model naturally completed the response (hit EOS or a stop sequence); `"length"` — hit `max_tokens` limit (response may be truncated; consider increasing max_tokens); `"tool_calls"` — model wants to call a tool (you must handle the tool call); `"content_filter"` — response filtered for safety. In production, always check `finish_reason`: `"length"` means your response was cut off and may be incomplete; `"tool_calls"` requires your code to handle the function call before proceeding.

**Q9: What is `tool_choice` in the OpenAI API and what are the options?**
> **Answer:** `tool_choice` controls whether and how the model uses tools. Options: `"auto"` (default) — model decides whether to call a tool based on the query; `"none"` — model never calls tools, always responds directly; `"required"` — model must call at least one tool; `{"type": "function", "function": {"name": "specific_tool"}}` — forces the model to call a specific named tool. Use `"required"` when your workflow depends on always getting structured function arguments; use `"auto"` for natural assistant behavior where tool use is optional.

**Q10: How do you maintain conversation history in a multi-turn chatbot application?**
> **Answer:** Maintain a `history` list and append each message (both user inputs and assistant responses) to it. On every API call, pass the full history in the `messages` array. Example pattern: `history.append({"role": "user", "content": user_msg})`, then call API, then `history.append({"role": "assistant", "content": response})`. The model uses the full history to understand context for each new message. Important consideration: conversation history grows with each turn, consuming more tokens and increasing cost. For long conversations, implement context window management (truncate old messages, use summarization).

---

## 14. Key Learning Thoughts {#learning-thoughts}

> **Thought 1 — Roles Are the Architecture of Your Application**
> When you design an LLM application, the first decision is: what goes in the system message? This is your "business logic layer." The system message is the most persistent and impactful part of your prompt. Get this right before worrying about anything else. A well-crafted system message can do 80% of the work.

> **Thought 2 — Tool Calling Is the Bridge Between LLMs and the Real World**
> Base LLMs have knowledge cutoffs and can't access real-time data or perform precise computation. Tool calling bridges this gap: let the LLM handle natural language understanding and let your code handle execution, data fetching, and computation. This division of labor produces more accurate, faster, and cheaper systems than trying to make the LLM do everything.

> **Thought 3 — The Token Budget Is Your Real Constraint**
> Every API call costs tokens. In production: system message (fixed cost per call), conversation history (grows with turns), user message (user-controlled), and output (your max_tokens cap). Profile your token usage with `.usage.total_tokens` and optimize accordingly. Tool calling's 71% token reduction in the LCM example isn't academic — at 10M requests/day, that's a massive cost difference.

> **Thought 4 — Conversation History Is Not Free**
> Building a "conversational" chatbot means sending the FULL conversation history to the API every single turn. A 10-turn conversation might send 5,000 tokens per call by turn 10. Design your history management strategy early: sliding window (keep last N turns), summarization (compress old history), or topic segmentation (start fresh on new topics).

> **Thought 5 — The Few-Shot Assistant Pattern Is Underrated**
> Most engineers know about few-shot prompting in the user message. Far fewer know you can use the `assistant` role to inject example responses. This technique is more powerful because: (1) it shows format AND content in one step; (2) the model directly mimics examples rather than interpreting instructions about format; (3) you can show edge case handling without lengthy instruction text.

> **Thought 6 — Always Check `finish_reason`**
> `"length"` means your response was cut short. If you're extracting structured data and finish_reason = "length", you may have malformed JSON or missing fields downstream. Add `assert response.choices[0].finish_reason == "stop"` or handle `"length"` gracefully in production.

> **Thought 7 — JSON Mode and Structured Outputs Remove a Whole Class of Bugs**
> A major source of production bugs in LLM apps is parsing failures — the model returns almost-valid JSON with a trailing comma, or uses the wrong key name. JSON mode eliminates JSON parsing failures. Structured Outputs with Pydantic eliminates schema validation failures. For any production code that parses LLM output, always use one of these two methods.

> **Thought 8 — Model Selection Is a Trade-off, Not a Competition**
> Don't always reach for the most powerful model. GPT-4o-mini at $0.60/1M output tokens is often the right choice for classification, simple QA, or structured extraction tasks where GPT-5.4 at 10× the cost would produce identical results. Match model capability to task complexity. Use the most capable model only when genuinely needed.

---

## Quick Reference — API Pattern Templates

### Pattern 1: Simple Q&A
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question}
    ],
    temperature=0.3,
    max_tokens=256
)
answer = response.choices[0].message.content
```

### Pattern 2: Multi-Turn Chatbot
```python
history = []
while True:
    user_input = input("You: ")
    history.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history
    )
    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    print(f"Assistant: {reply}")
```

### Pattern 3: Structured Data Extraction
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract information and return valid JSON with keys: name, date, location."},
        {"role": "user", "content": f"Text: {document}"}
    ],
    response_format={"type": "json_object"},
    temperature=0.0
)
data = json.loads(response.choices[0].message.content)
```

### Pattern 4: Tool Calling
```python
# Step 1: Define tools + first call
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)
# Step 2: Handle tool call
if response.choices[0].finish_reason == "tool_calls":
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    result = your_function(**args)
    # Step 3: Second call with result
    messages.append(response.choices[0].message)
    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})
    final = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
```

---

## Post-Read Resources
- OpenAI API Documentation: https://developers.openai.com/api/docs/quickstart
- OpenAI Function Calling Guide: https://platform.openai.com/docs/guides/function-calling
- Anthropic Prompt Engineering: https://github.com/anthropics/prompt-eng-interactive-tutorial/
- HuggingFace Inference Guide: https://huggingface.co/docs/huggingface_hub/guides/inference
