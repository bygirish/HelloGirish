# Section 2: Large Language Models — Basic Prompting
**Lecture 4 | IIT Kharagpur × upGrad | Instructor: Prof. Sourangshu Bhattacharya**

> **Learning Goal:** Understand what a prompt is, how to structure one using its four elements, how to write a basic API call, and how to construct sophisticated role-based prompts. Be able to write your first LLM-powered Python script confidently.

---

## Table of Contents
1. [What Are Prompts?](#what-are-prompts)
2. [Prompt Engineering as a Skill](#prompt-engineering-skill)
3. [Your First Basic Prompt — API Call Walkthrough](#first-prompt)
4. [A More Advanced Prompt — The Socratic Debugger](#advanced-prompt)
5. [Elements of a Prompt](#elements)
6. [Prompt Design Principles](#principles)
7. [Interview Questions](#interview)
8. [Key Learning Thoughts](#learning-thoughts)

---

## 1. What Are Prompts? {#what-are-prompts}

A **prompt** is the instructions and context you pass to a language model to get a desired output.

More precisely:
> A prompt involves **instructions and context** passed to a language model to achieve a desired task.

Think of it as the "input specification" for an LLM. Just as a function in programming takes arguments and returns a value, a prompt is the argument you pass to an LLM to get its output.

### The Old Mental Model vs The New One

| Old (pre-GPT-3) | New (Prompting Era) |
|---|---|
| Train a model for each task | One general model, different prompts |
| Task-specific fine-tuning required | No gradient updates needed |
| Need labeled datasets per task | Natural language instructions suffice |
| Model switching for each use case | Single API, many applications |

### Why Prompting Matters Now
Before GPT-3, AI engineers had to fine-tune a separate model for each task (sentiment: one model, translation: another model, summarization: yet another). Now, a single foundation model can do all of these — if you know how to write the right prompt.

---

## 2. Prompt Engineering as a Skill {#prompt-engineering-skill}

**Prompt engineering** is the practice of developing and optimizing prompts to efficiently use language models for a variety of applications.

> It is a useful skill for AI engineers and researchers to improve and efficiently use language models.

### What Prompt Engineering Is NOT
- It is NOT magic — it's a systematic, learnable craft
- It is NOT just "being nice to the AI"
- It is NOT about memorizing a fixed formula

### What Prompt Engineering IS
- Communicating your intent clearly and precisely to a model
- Understanding the model's strengths and limitations
- Iteratively testing and refining instructions
- Structuring context and examples to guide output format and quality

### The Difference Between a Naive User and a Prompt Engineer
**Naive user:** "Summarize this article."
**Prompt engineer:** "Summarize the following article in 3 bullet points. Each bullet should be a single sentence. Focus on the main argument, supporting evidence, and conclusion. Article: [text]"

Both get a summary. The second gets a *reliable, formatted, task-specific* summary every time.

---

## 3. Your First Basic Prompt — API Call Walkthrough {#first-prompt}

### The Simplest Possible Prompt
Input: `"The sky is"` → Model continues: `"The sky is a beautiful blue color during the day..."`

This is completion: the model predicts the most likely continuation. This is how base models work at their core.

### First API Call (OpenAI)

```python
from openai import OpenAI

# client reads OPENAI_API_KEY from environment variable
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "The sky is"}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
```

### Breaking Down the Call

| Parameter | What it does | Example value |
|---|---|---|
| `model` | Which LLM to use | `"gpt-4o-mini"` |
| `messages` | The conversation history + current prompt | List of role/content dicts |
| `temperature` | Controls randomness of output | `0.7` |
| `max_tokens` | Hard cap on output length | `200` |

### The Response Object
The API returns a structured JSON response. You access the generated text via:
```python
response.choices[0].message.content
```
- `choices[0]` — the first (and usually only) completion
- `.message.content` — the actual generated text string

---

## 4. A More Advanced Prompt — The Socratic Debugger {#advanced-prompt}

This example from the lecture shows the real power of prompt engineering. Instead of asking the model to "fix a bug," you can give it a **persona, rules, and behavioral constraints**.

### The Prompt

```python
messages = [
    {
        "role": "system",
        "content": """You are a Socratic Debugger. Never fix code directly — guide 
the user to find the bug themselves through questions and analogies.

Rules:
1. Don't reveal the bug until the user says "give up" or asks 3+ times.
2. Ask what they *think* the code does vs. what it actually does.
3. Respond only with probing questions or nudging analogies.
4. End each reply with [Proximity: X/10] — how close they are.
5. At proximity 9+, give one direct hint. Celebrate when they find it!"""
    },
    {
        "role": "user",
        "content": """My Python function returns wrong averages — why?

def average(nums):
    total = 0
    for n in nums:
        total =+ n
    return total / len(nums)

print(average([10, 20, 30]))  # Expected: 20.0, Got: 30.0"""
    }
]
```

### What Makes This Prompt "Advanced"

| Feature | What it does |
|---|---|
| **Persona assignment** | "You are a Socratic Debugger" — sets behavior mode |
| **Explicit prohibitions** | "Never fix code directly" — prevents default helpful behavior |
| **Numbered rules** | Precise, ordered constraints the model will follow |
| **Custom output format** | `[Proximity: X/10]` — structured signal in every response |
| **Conditional behavior** | "At proximity 9+, give one direct hint" — logic branching |
| **Emotional tone** | "Celebrate when they find it!" — shapes the interaction feel |

### The Output
The model responds with:
> "What do you think the line `total =+ n` is doing in your function? Does it look like it's adding `n` to `total` as you might expect?
> Is there a difference between the operation you intended and what is actually happening in that line?
> [Proximity: 4/10]"

Notice: the model correctly identified the bug (`=+` vs `+=`) but **refuses to state it directly** — it guides through questions exactly as instructed.

### The Lesson
This example proves that prompts are **programs**. They encode logic, rules, personas, and output formats. The model is the runtime; the prompt is the code.

---

## 5. Elements of a Prompt {#elements}

Every well-structured prompt is composed of up to four elements. Not all are always present, but understanding all four lets you build precise, reliable prompts.

### The Four Elements

```
┌─────────────────────────────────────────────────────────┐
│  INSTRUCTION    Classify the text into neutral,         │
│                 negative or positive                    │
│                                                         │
│  CONTEXT        (optional background: e.g., this is     │
│                 a product review for a restaurant)      │
│                                                         │
│  INPUT DATA     Text: I think the food was okay.        │
│                                                         │
│  OUTPUT         Sentiment:                              │
│  INDICATOR                                              │
└─────────────────────────────────────────────────────────┘
```

### Element 1: Instructions
**What it is:** The task directive — what you want the model to do.

**Examples:**
- `"Classify the sentiment of the following text."`
- `"Summarize the article in 3 bullet points."`
- `"Write a Python function that takes a list and returns sorted unique elements."`
- `"Translate the following English text to French."`

**Best practices:**
- Be explicit and specific — `"Summarize in 3 bullet points"` >> `"Summarize"`
- Use action verbs: Classify, Summarize, Translate, Write, Explain, Extract
- State format requirements upfront: `"Respond in JSON with keys 'name' and 'score'"`

### Element 2: Context
**What it is:** Background information that helps the model understand the situation, domain, or constraints.

**Without context:**
```
Classify the sentiment.
Text: The battery lasts forever and the screen is gorgeous.
```

**With context:**
```
You are analyzing customer reviews for a smartphone product page.
The reviews should be classified to determine which features to highlight.
Classify the sentiment.
Text: The battery lasts forever and the screen is gorgeous.
```

Context matters most when:
- The domain has specialized vocabulary (medical, legal, financial)
- The model's default behavior doesn't match your use case
- You need the model to maintain a specific persona

### Element 3: Input Data
**What it is:** The actual data the model should process — the text to classify, the code to debug, the document to summarize.

**Formatting tips:**
- Clearly label your input: `Text: ...`, `Code: ...`, `Article: ...`
- For long inputs, use delimiters: triple backticks, XML-style tags
- For structured data, use a consistent format

```python
# Good: labeled and delimited
prompt = """
Summarize the following customer support ticket.

Ticket:
\"\"\"
The user reports that after updating to version 3.2.1, 
the app crashes immediately on launch. Issue occurs on 
both iOS 16 and iOS 17. Previously working on 3.2.0.
\"\"\"

Summary:
"""
```

### Element 4: Output Indicator
**What it is:** A signal that tells the model what format to produce, often by partially starting the output.

**Examples:**
- `Sentiment:` → model fills in: `Positive`
- `SQL Query:` → model fills in SQL
- `{"name":` → model continues the JSON
- `def calculate_` → model writes a function starting with that name

**Why it works:** The model is trained to continue patterns. Giving it the start of the output format strongly steers the format of the completion.

### Worked Example — All Four Elements Together

```
[INSTRUCTION]
Classify the following customer message into one of these categories:
billing_issue, technical_problem, general_inquiry, complaint.

[CONTEXT]
You are classifying tickets for a SaaS customer support team.
Respond only with the category label, nothing else.

[INPUT DATA]
Message: "I've been charged twice this month and my account shows 
the same invoice number for both transactions."

[OUTPUT INDICATOR]
Category:
```

Expected output: `billing_issue`

---

## 6. Prompt Design Principles {#principles}

### Principle 1: Be Specific, Not Vague
| Vague | Specific |
|---|---|
| "Write something about AI" | "Write a 150-word blog introduction about how AI is changing healthcare diagnostics" |
| "Fix my code" | "Find and fix the off-by-one error in the following Python function" |
| "Summarize this" | "Summarize in 3 bullet points, each under 15 words, focusing on key decisions" |

### Principle 2: Use Examples (Few-Shot)
When the task is ambiguous or the output format is complex, show examples:

```
Convert informal messages to professional email tone.

Informal: "hey, can u send me that doc asap?"
Professional: "Could you please share the document at your earliest convenience?"

Informal: "this meeting is pointless lol"
Professional: "I feel the meeting agenda could be better defined to improve productivity."

Informal: "the client is being super annoying about the deadline"
Professional:
```

### Principle 3: Use Delimiters to Separate Content
Use `"""`, `---`, `###`, or XML tags to clearly separate instructions from content:

```python
prompt = f"""
Analyze the sentiment of the text between the triple backticks.
Respond with exactly one word: Positive, Negative, or Neutral.

Text:
```{user_text}```
"""
```

### Principle 4: Specify Output Format Explicitly
```
Extract the following information from the resume and return as JSON:
- name (string)
- years_of_experience (integer)  
- top_skills (list of strings, max 5)

Resume: [text]

JSON:
```

### Principle 5: Assign a Persona for Consistent Tone
```
You are an expert Python developer who writes clean, well-documented code.
When explaining concepts, use simple analogies before technical details.
Always consider edge cases and mention them explicitly.
```

### Principle 6: Chain of Thought for Complex Reasoning
For multi-step problems, ask the model to "think step by step":
```
Solve the following word problem. Think step by step before giving the final answer.

Problem: A store offers 20% discount on all items. If an item costs $85 before 
discount, and there's 8% tax on the discounted price, what is the final price?

Step-by-step solution:
```

---

## 7. Interview Questions {#interview}

**Q1: What are the four elements of a well-structured prompt?**
> **Answer:** (1) **Instructions** — the task directive (what to do); (2) **Context** — background information that shapes how the task should be done; (3) **Input Data** — the actual content to process; (4) **Output Indicator** — a signal that steers the model's output format, often by partially initiating the expected output. Not all four are always required — simple tasks may only need instructions and input data. But knowing all four lets you build reliable prompts for complex tasks.

**Q2: What is prompt engineering and why is it important?**
> **Answer:** Prompt engineering is the practice of developing and optimizing prompts to efficiently use language models for a variety of applications. It's important because: (1) foundation models are general-purpose — the prompt is how you specialize them for your task; (2) the same model can produce dramatically different quality outputs depending on prompt quality; (3) it's a core skill for AI engineers to avoid costly fine-tuning for tasks that can be solved with better prompting; (4) prompts encode logic, personas, constraints, and output formats — treating them as "programs" leads to more reliable systems.

**Q3: What is the difference between a "system" message and a "user" message in a chat completion API?**
> **Answer:** The **system** message sets the overall behavior, tone, persona, and constraints for the assistant — it's read first and persists as the model's "instructions" for the conversation. Example: `"You are a helpful assistant that only answers questions about Python."` The **user** message represents the actual human input — the question or instruction from the person using the system. The key distinction: system message = *how the model should behave*; user message = *what the user wants right now*.

**Q4: Why does adding an "output indicator" like "Sentiment:" at the end of a prompt help?**
> **Answer:** Language models are trained to continue patterns. By providing the start of the expected output format, you leverage the model's completion instinct to produce output in that exact format. `"Sentiment:"` after a classification prompt practically guarantees the model writes a sentiment label next, not a full paragraph. This is especially powerful for structured outputs: starting `{"name":` steers the model to produce valid JSON; starting `SELECT` steers it to produce SQL.

**Q5: What makes the Socratic Debugger prompt "advanced"? What techniques does it demonstrate?**
> **Answer:** It demonstrates: (1) **Persona assignment** — giving the model a specific role changes its default helpful behavior; (2) **Explicit prohibitions** — "Never fix code directly" overrides default behavior; (3) **Rule-based constraints** — numbered rules give precise, ordered behavioral specifications; (4) **Custom output format** — `[Proximity: X/10]` enforces structured information in every response; (5) **Conditional logic** — "At proximity 9+, give one direct hint" encodes branching behavior; (6) **Tone control** — "Celebrate when they find it!" shapes the emotional register.

**Q6: When would you prefer few-shot prompting over zero-shot?**
> **Answer:** Few-shot prompting is preferred when: (1) the output format is complex or unusual (JSON with specific keys, structured tables); (2) the task has a specific style that's hard to describe but easy to show; (3) zero-shot gives inconsistent results; (4) the task requires a specific mapping that isn't in the model's default behavior (e.g., domain-specific label names like "BILLING_DISPUTE" instead of "billing issue"). Zero-shot is preferred when the task is standard and well-defined, or when token budget is a concern.

**Q7: What is the difference between a prompt and a fine-tuned model?**
> **Answer:** A **prompt** is runtime instruction — it changes how the model behaves for that specific call without altering the model's weights. It's cheap, flexible, and instant to change. A **fine-tuned model** has its weights updated on task-specific data — the behavior change is permanent and deeper. Fine-tuning wins when: you need consistent style/format across thousands of calls; the task requires knowledge not in the base model; prompt-based approaches have hit a ceiling. Prompting wins when: the task is well-defined; you need flexibility; you can't afford fine-tuning compute.

---

## 8. Key Learning Thoughts {#learning-thoughts}

> **Thought 1 — A Prompt is a Program**
> When you write a complex prompt with rules, personas, output formats, and conditional logic — you are programming. The model is the runtime. This mental shift is crucial: stop thinking of prompts as "talking to AI" and start thinking of them as writing specifications. The Socratic Debugger example executes a multi-step pedagogical algorithm via natural language.

> **Thought 2 — The Output Indicator is Underused**
> Most beginners write instructions and input but forget the output indicator. Simply ending your prompt with `"Summary:"`, `"Category:"`, `"Python code:"`, or `{"` drastically improves format consistency. It's the cheapest reliability improvement available.

> **Thought 3 — Context Shapes Interpretation**
> The same input can have entirely different correct outputs depending on context. "The bank was steep" is about rivers with hiking context, about finance with banking context. Always provide the context that resolves ambiguity for your use case.

> **Thought 4 — Specificity Scales with Reliability**
> The more specific your prompt, the more reliable your outputs. Vague prompts work when you're exploring; specific prompts work when you're building production systems. As you move from prototype to production, always tighten the prompt.

> **Thought 5 — Prompt Engineering is Empirical**
> Unlike writing code that is deterministic, prompt design requires experimentation. You write a prompt, test it on diverse inputs, observe failures, and refine. Build a small test set of inputs, including edge cases, and iterate until the prompt handles them all well. This is the prompt engineering workflow.

> **Thought 6 — Few-Shot Examples are More Powerful Than Lengthy Instructions**
> If your instructions are getting long and complicated, try showing examples instead. A model that struggles to follow "respond in a formal, professional tone with bullet points, avoiding passive voice" will often immediately mimic a clear example of that exact style. Show, don't just tell.

> **Thought 7 — The API is Just the Delivery Mechanism**
> The `client.chat.completions.create()` call is just infrastructure. The real intellectual work is in the `messages` array. Focus most of your design effort on crafting the system message and structuring the user message well — the API parameters (temperature, max_tokens) are tuning knobs that come later.

---

## Quick Reference — Prompt Template

```python
from openai import OpenAI
client = OpenAI()

system_prompt = """
[PERSONA/ROLE]
You are [role description].

[CONSTRAINTS/RULES]
1. [Rule 1]
2. [Rule 2]

[OUTPUT FORMAT]
Respond in the following format:
[format specification]
"""

user_prompt = """
[CONTEXT (if needed)]
[context]

[INSTRUCTION]
[what to do]

[INPUT DATA]
[actual data]

[OUTPUT INDICATOR]
[start of expected output]:
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.3,    # low for factual tasks
    max_tokens=500
)

print(response.choices[0].message.content)
```

---

## Post-Read Resources
- Prompt Engineering Guide: https://www.promptingguide.ai/
- Anthropic Prompt Engineering Tutorial: https://github.com/anthropics/prompt-eng-interactive-tutorial/
- OpenAI Quickstart: https://developers.openai.com/api/docs/quickstart
