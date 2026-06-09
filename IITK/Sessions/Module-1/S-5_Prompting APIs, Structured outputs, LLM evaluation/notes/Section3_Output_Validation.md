# Section 3 — Structured Outputs: Output Validation
### Lecture 5 · IIT Kharagpur · Prof. Sourangshu Bhattacharya

---

## What This Section Covers

| Topic | Slide Range |
|-------|-------------|
| Why LLMs don't always return clean JSON | 44 |
| Pydantic: schema validation for Python | 46 |
| `@dataclass` vs `BaseModel` — the critical difference | 47 |
| `ValidationError` — catching LLM mistakes early | 48 |
| The naive approach: asking for JSON in text | 49 |
| Tool calling forces structured output | 50 |
| `model_json_schema()` — auto-generated schemas | 51 |
| Instructor: the cleanest API | 52 |
| Nested models for complex structures | 53–54 |
| Prompting tips for Pydantic models | 55 |

---

## Part 3A — The Problem: LLMs Don't Return Clean Data

### The Core Challenge

When you ask an LLM to "return JSON," it is a polite request, not a guarantee. LLMs are trained to produce human-readable text, and their training data is full of:
- JSON wrapped in markdown code blocks
- JSON with trailing prose explanations
- JSON with incorrect types (age returned as `"25"` instead of `25`)
- JSON with missing fields
- Completely hallucinated field values

**Common failure modes:**

**Failure 1: Markdown code block wrapping**
```python
# What you asked for:
{"name": "Alice", "age": 25}

# What the model returns:
"""
```json
{"name": "Alice", "age": 25}
```
"""
```
Your `json.loads()` crashes on the backticks.

**Failure 2: Type mismatch**
```python
# Model returns:
{"name": "Alice", "age": "25"}  # age is a string, not an int

# Downstream crash:
person.age + 1   # TypeError: can only concatenate str (not "int") to str
```

**Failure 3: Missing field**
```python
# Your schema expects:
{"name": str, "age": int, "email": str}

# Model returns:
{"name": "Alice", "age": 25}  # 'email' missing
```

**Failure 4: Hallucinated field value**
```python
# Context says: "age: unknown"
# Model returns: {"age": "thirty-something"}  # not an int at all
```

The goal of this section: **build a reliable pipeline** that catches all these failures systematically, before they propagate into your application.

---

## Part 3B — Pydantic: Schema Validation for Python

### What is Pydantic?

Pydantic is a Python library that defines data models with **type annotations** and **validates** data against them at parse time. Unlike `@dataclass`, Pydantic does not trust that the data is correct — it verifies and coerces.

```bash
pip install pydantic
```

---

### `@dataclass` vs `BaseModel` — The Critical Difference

This is one of the most important distinctions in modern Python data engineering.

```python
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError

# === @dataclass: accepts wrong types SILENTLY ===
@dataclass
class PersonDC:
    name: str
    age: int

# LLM returns age as a string — dataclass silently accepts it
dc = PersonDC(name="Sam", age="25")
print(type(dc.age))   # <class 'str'>  ← should be int!

# Crash happens LATER, far from the source:
dc.age + 1   # TypeError: can only concatenate str (not "int") to str

# === Pydantic BaseModel: validates at construction time ===
class PersonPD(BaseModel):
    name: str
    age: int

# Case 1: Coercible string → coerced to int automatically
pd = PersonPD(name="Sam", age="25")
print(type(pd.age))   # <class 'int'>  ← coerced! ✅
print(pd.age + 1)     # 26 ✅

# Case 2: Non-coercible value → immediate, clear error
try:
    bad = PersonPD(name="Sam", age="twenty-five")
except ValidationError as e:
    print("ValidationError caught at parse time!")
    for err in e.errors():
        print(f"  Field: {err['loc'][0]}")
        print(f"  Problem: {err['msg']}")
        print(f"  Value: {err['input']}")
```

**The key insight:** With `@dataclass`, the bug from a bad LLM response surfaces far from the point of failure — possibly in a different function, a different file, even a different service. With Pydantic, the failure is **loud, immediate, and informative** — right at the point where the LLM response is parsed.

| Feature | `@dataclass` | `BaseModel` (Pydantic) |
|---------|-------------|------------------------|
| Type checking at construction | ❌ None | ✅ Always |
| Type coercion (`"10"` → `10`) | ❌ Silent failure | ✅ Automatic |
| Error on bad input | ❌ Silent | ✅ `ValidationError` with details |
| JSON Schema generation | ❌ Manual | ✅ `model_json_schema()` |
| Nested model support | ⚠️ Manual | ✅ Automatic with `$defs/$ref` |
| Serialization | `dataclasses.asdict()` | `model.model_dump()` |
| JSON parsing | Manual | `Model.model_validate_json(str)` |

---

### Type Coercion — Pydantic's Superpower for LLM Output

Pydantic attempts to coerce values to the declared type before raising an error. This handles the most common LLM mistake: returning numbers as strings.

```python
from pydantic import BaseModel
from typing import Optional
import datetime

class Event(BaseModel):
    name: str
    year: int
    score: float
    active: bool
    date: Optional[datetime.date]

# All strings — typical LLM JSON output:
e = Event(
    name="Conference",
    year="2024",       # str → int ✅
    score="9.5",       # str → float ✅
    active="true",     # str → bool ✅
    date="2024-06-15"  # str → datetime.date ✅
)
print(type(e.year))   # <class 'int'>
print(type(e.date))   # <class 'datetime.date'>
```

> **Learning Thought:** This coercion behavior is not magic — Pydantic uses Python's type system and a set of coercion rules. The boundary is clear: Pydantic coerces "reasonable" transformations (string digits to int, ISO strings to dates). It raises `ValidationError` for "unreasonable" ones (arbitrary strings to int, non-boolean strings to bool). Know the boundary.

---

### `ValidationError` — The Single Most Important Feature

When validation fails, Pydantic raises a detailed `ValidationError` (not a cryptic `TypeError` three call-frames later):

```python
from pydantic import BaseModel, ValidationError

class Person(BaseModel):
    name: str
    age: int

try:
    Person(name="Sam", age="not_a_number")
except ValidationError as e:
    print(e)
```

Output:
```
1 validation error for Person
age
  Input should be a valid integer, unable to parse string as an integer
  [type=int_parsing, input_value='not_a_number', input_url=...]
```

You get:
- **Which model** failed
- **Which field** failed
- **Why** it failed (exact error type)
- **What the value was** (helps debug LLM hallucinations)

---

### `model_json_schema()` — Auto-generating JSON Schemas

This is one of the most practically important features for LLM integration. Instead of writing 30+ lines of JSON Schema by hand (which is error-prone and becomes stale when you rename a field), Pydantic generates it automatically.

```python
from pydantic import BaseModel, Field
from typing import List, Optional
import json

class PythonPackage(BaseModel):
    name: str     = Field(description="Package name")
    author: str   = Field(description="Primary author or organization")
    version: Optional[str] = Field(None, description="Latest stable version if known")
    purpose: str  = Field(description="One-sentence description of the package")

class PackageList(BaseModel):
    packages: List[PythonPackage]

# Auto-generates complete JSON Schema:
schema = PackageList.model_json_schema()
print(json.dumps(schema, indent=2))
```

Output (auto-generated, zero maintenance):
```json
{
  "title": "PackageList",
  "type": "object",
  "properties": {
    "packages": {
      "items": {"$ref": "#/$defs/PythonPackage"},
      "title": "Packages",
      "type": "array"
    }
  },
  "$defs": {
    "PythonPackage": {
      "properties": {
        "name": {"description": "Package name", "title": "Name", "type": "string"},
        "author": {"description": "Primary author...", "type": "string"},
        "version": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
        "purpose": {"description": "One-sentence description...", "type": "string"}
      },
      "required": ["name", "author", "purpose"]
    }
  }
}
```

> **Learning Thought:** `model_json_schema()` is the contract between your Python code and the LLM. When you rename a field in Python, the schema updates automatically — no drift, no stale documentation. This is the right engineering practice.

---

## Part 3C — Three Approaches to Getting Structured Output

### Approach 1 (Naive): Ask for JSON in the Prompt

```python
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": "Return JSON with fields: name, age."},
        {"role": "user",   "content": "Tell me about Alice, age 25."}
    ]
)
raw = response.choices[0].message.content
# Might be: '{"name": "Alice", "age": 25}' ✅
# Might be: '```json\n{"name": "Alice", "age": 25}\n```' ❌
# Might be: 'Here is the JSON:\n{"name": "Alice", "age": 25}' ❌
```

**Why it's fragile:** No enforcement mechanism. The model has learned to be "helpful" by adding explanations and code blocks. The more natural the model's training, the more likely it wraps output in prose.

---

### Approach 2: Tool Calling with `model_json_schema()`

OpenAI's `tools` parameter accepts a JSON Schema and **forces** the model to produce output conforming to it. But writing JSON Schema by hand is painful:

```python
# BAD: Hand-written schema (painful, error-prone, becomes stale):
tools = [{
    "type": "function",
    "function": {
        "name": "extract_person",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "age": {"type": "integer", "description": "Person's age"}
            },
            "required": ["name", "age"]
        }
    }
}]

# GOOD: Auto-generated schema from Pydantic:
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int  = Field(description="Person's age")

tools = [{
    "type": "function",
    "function": {
        "name": "extract_person",
        "description": "Extract person information",
        "parameters": Person.model_json_schema()  # ← auto-generated!
    }
}]

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Alice is 25 years old."}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_person"}}
)

# Parse and validate the response
raw_args = response.choices[0].message.tool_calls[0].function.arguments
result   = Person.model_validate_json(raw_args)
print(result.name, result.age)  # Alice 25
print(type(result.age))          # <class 'int'>  ← guaranteed by Pydantic
```

**This approach:**
1. LLM is forced to call the function (structured output guaranteed by API)
2. Pydantic validates the types (hallucination caught at parse time)
3. Schema stays in sync with your Python model (zero maintenance)

---

### Approach 3: Instructor — The Cleanest API

`instructor` wraps any OpenAI-compatible client and handles the entire pipeline automatically:
- Schema generation from Pydantic model
- Tool call setup
- Response parsing
- Type validation
- **Automatic retries** if validation fails

```bash
pip install instructor
```

**Without instructor (verbose):**
```python
# ~15 lines every time
tools = [{"type": "function", "function": {"name": "...", "parameters": Model.model_json_schema()}}]
response = client.chat.completions.create(model=..., messages=..., tools=tools, tool_choice=...)
raw_args = response.choices[0].message.tool_calls[0].function.arguments
result   = Model.model_validate_json(raw_args)
```

**With instructor (3 lines every time):**
```python
import instructor
from openai import OpenAI

# Wrap the client once:
client = instructor.from_openai(OpenAI())

# One extra parameter — that's all:
result = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Tell me about pydantic and fastapi."}],
    response_model=PackageList,   # ← This is the entire change
)

# result IS a PackageList object — fully typed, validated, ready to use
print(isinstance(result, PackageList))   # True
print(isinstance(result.packages[0], PythonPackage))  # True
```

**Groq with instructor:**
```python
import instructor
from groq import Groq

client = instructor.from_groq(
    Groq(api_key="..."),
    mode=instructor.Mode.JSON,  # Groq works best with JSON mode
)

result = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "..."}],
    response_model=PackageList,
)
```

> **Learning Thought:** The pattern `response_model=YourPydanticModel` is the key to understanding instructor. You declare the output type in Python, and instructor makes the LLM conform to it. This is the same mental model as type hints — you express intent, the library enforces it.

---

## Part 3D — Nested Models for Complex Structures

### Why Nested Models?

Real-world LLM tasks return hierarchical data. Flat fields lose semantic relationships:

```python
# Flat (bad): relationship between dates is implicit
class SearchQuery(BaseModel):
    start_date: str
    end_date: str
    query: str

# Nested (good): DateRange groups related fields explicitly
class DateRange(BaseModel):
    start: datetime.date
    end: datetime.date

class SearchQuery(BaseModel):
    rewritten_query: str
    published_daterange: DateRange   # ← nested!
    domains_allow_list: List[str]
```

The nested schema auto-generates `$defs/$ref` entries in JSON Schema — OpenAI tool calling handles this natively.

---

### Real Example: Semantic Search Query Understanding

A search engine that rewrites user queries into structured API payloads — without the user knowing the internal structure:

```python
import instructor
import datetime
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI

class DateRange(BaseModel):
    start: datetime.date = Field(description="Start date of the search range (ISO 8601)")
    end: datetime.date   = Field(description="End date of the search range (ISO 8601)")

class SearchQuery(BaseModel):
    rewritten_query: str = Field(
        description="Expanded, keyword-rich version of the user query"
    )
    published_daterange: DateRange = Field(
        description="Publication date range to filter results"
    )
    domains_allow_list: List[str] = Field(
        description="Trusted domains to restrict search to"
    )

client = instructor.from_openai(OpenAI())

def understand_query(user_input: str) -> SearchQuery:
    return client.chat.completions.create(
        model="gpt-4.1",
        response_model=SearchQuery,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a search query understanding system. "
                    f"Today's date is {datetime.date.today()}. "
                    "Rewrite queries into structured search payloads."
                )
            },
            {"role": "user", "content": user_input}
        ],
    )

# User types natural language:
result = understand_query("recent advancements in AI safety")

# You get a fully structured, typed Python object:
print(result.rewritten_query)
print(result.published_daterange.start, "→", result.published_daterange.end)
print(result.domains_allow_list)

# Ready to call your search API:
import json
print(json.dumps(result.model_dump(mode="json"), indent=2))
```

Output:
```
"AI safety alignment research advancements 2024 2025"
2024-01-01 → 2025-12-31
['arxiv.org', 'openai.com', 'deepmind.com']
```

> **Learning Thought:** The user sees: a search box. Your system sees: a structured `SearchQuery` object with typed fields ready to call a search API. The LLM is the bridge. Pydantic + instructor is the guarantor of correctness between the natural language world and the typed code world.

---

### Serialization with Nested Models

```python
# model_dump() → dict (all nested models recursively flattened)
result.model_dump()
# {'rewritten_query': '...', 'published_daterange': {'start': date(...), 'end': date(...)}, ...}

# model_dump(mode="json") → JSON-serializable dict (dates become strings)
result.model_dump(mode="json")
# {'published_daterange': {'start': '2024-01-01', 'end': '2025-12-31'}, ...}

# model_validate_json(str) → constructs nested model from JSON string
reconstructed = SearchQuery.model_validate_json(json_string)
# Automatically constructs DateRange from nested dict, converts strings to dates
```

---

### Advanced Pattern: Enum Fields + Optional Fields

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Sentiment(str, Enum):
    positive = "positive"
    neutral  = "neutral"
    negative = "negative"

class AspectSentiment(BaseModel):
    aspect: str = Field(description="Product aspect (e.g., 'battery life')")
    sentiment: Sentiment
    quote: Optional[str] = Field(None, description="Supporting quote from review")

class ReviewAnalysis(BaseModel):
    overall_sentiment: Sentiment
    rating_prediction: Optional[int] = Field(None, description="Predicted 1-5 stars")
    aspect_sentiments: List[AspectSentiment]
    key_complaints: List[str]
    key_praises: List[str]
    would_recommend: Optional[bool] = None
```

**Why `Optional[...]` fields matter for LLMs:**
- The LLM may not always be able to infer the value (e.g., `rating_prediction` when the review text is ambiguous)
- `Optional[str] = Field(None, ...)` means: if the model can't determine this, return `null` (valid JSON)
- Pydantic marks it as not required in the schema — LLM won't hallucinate a fake value

**Why `str` Enum vs plain `str`:**
- `Sentiment(str, Enum)` with explicit values constrains the LLM to exactly `"positive"`, `"neutral"`, or `"negative"`
- A plain `str` field would accept `"very positive"`, `"mixed"`, `"somewhat negative"` — hard to handle downstream
- Pydantic validates the enum value, catching LLM drift

---

## Part 3E — Prompting Tips for Pydantic Models

Well-designed Pydantic models **self-document** the expected output. The schema IS the prompt (when using tool calling or instructor). A few tips:

### Tip 1: Use Descriptive Field Names

Field names become part of the JSON Schema and are visible to the model. Choose names that communicate intent:

```python
# Bad — model must guess what 'q' means:
class SearchQuery(BaseModel):
    q: str
    dr: dict

# Good — model reads field names as semantic hints:
class SearchQuery(BaseModel):
    rewritten_query: str
    published_daterange: DateRange
```

### Tip 2: Add `Field(description=...)`

Descriptions appear in the schema under the `description` key. The LLM reads them as instructions for what to put in that field:

```python
class Person(BaseModel):
    name: str = Field(description="Full legal name of the person")
    age: int  = Field(description="Age in years as an integer, not a string")
    email: str = Field(description="Primary email address in standard format user@domain.com")
```

### Tip 3: Use Nested Models Instead of Flat Dicts

```python
# Flat dict (model might put anything in):
class Event(BaseModel):
    date_info: dict  # could be {"year": 2024} or {"date": "2024-01-01"} or anything

# Typed nested model (Pydantic validates the internal structure too):
class DateInfo(BaseModel):
    year: int
    month: int  # 1-12
    day: int    # 1-31

class Event(BaseModel):
    date_info: DateInfo  # validated recursively
```

### Tip 4: Use `Optional[...]` for Inferable Fields

```python
class ProductReview(BaseModel):
    overall_sentiment: Sentiment              # always required
    rating: Optional[int] = None              # optional: might not be in text
    reviewer_age_group: Optional[str] = None  # optional: might not be mentioned
```

### Tip 5: Use `List[str]` for Variable-Length Outputs

```python
class Article(BaseModel):
    key_points: List[str]      # let the model decide how many
    tags: List[str]             # variable-length is fine
    references: List[str]       # model fills in as many as it finds
```

---

### True/False: Pydantic vs Instructor (Q&A from Lecture)

**Q: Which of the following statements is true?**
- ~~Pydantic is only used for output validation~~ → False. Pydantic also generates JSON schemas (`model_json_schema()`), handles serialization (`model_dump()`), and parses JSON (`model_validate_json()`). It's a complete data modeling library.
- ~~Instructor does schema validation~~ → False. Instructor delegates validation to Pydantic. Instructor handles the API integration (schema generation, tool calling, retry logic). Validation is Pydantic's job.
- **Pydantic can generate and validate schema** → True. This is its core dual capability.

---

## Interview Questions — Section 3

### Conceptual

**Q1. What are the two core failure modes when asking an LLM to "return JSON" in a plain message? How does each break downstream code?**

> **Failure 1 — Markdown wrapping:** The model wraps JSON in ` ```json ` fences (its training included code blocks). `json.loads()` raises `JSONDecodeError` immediately. **Failure 2 — Type mismatch:** The model returns `"age": "25"` (string) where you declared `age: int`. `json.loads()` succeeds but you get a string. Downstream: `person.age + 1` raises `TypeError`. The crash is silent and far from the source — worst kind of bug.

**Q2. Explain the difference between `@dataclass` and Pydantic `BaseModel` in the context of LLM output parsing.**

> `@dataclass` is Python's built-in data container — it stores values with type annotations but does NOT check or coerce types at runtime. If you pass `age="25"`, it stores `"25"` silently. `BaseModel` (Pydantic) validates and coerces at construction time: `age="25"` becomes `age=25` (int). If coercion is impossible (e.g., `age="twenty-five"`), it immediately raises a detailed `ValidationError`. For LLM output, which is inherently untrusted and may have type errors, Pydantic catches mistakes at the parse boundary before they propagate.

**Q3. What is `model_json_schema()` and why is it preferable to hand-writing JSON Schema for tool calling?**

> `model_json_schema()` is a Pydantic method that introspects the model's type annotations and `Field(description=...)` metadata to generate a complete, valid JSON Schema dict. It handles `Optional[...]` (nullable fields), `List[...]` (array types), nested models (`$defs/$ref`), and Enum constraints automatically. It's preferable to hand-writing because: (1) it never goes stale — rename a Python field and the schema updates; (2) zero bugs from typos in JSON; (3) handles complex nesting that is easy to get wrong by hand; (4) is a single function call per model class.

**Q4. What does the `instructor` library actually do under the hood?**

> `instructor` wraps an OpenAI-compatible client (OpenAI, Azure, Groq, Anthropic). When you call `client.chat.completions.create(..., response_model=MyModel)`, it: (1) calls `MyModel.model_json_schema()` to generate the tool schema; (2) adds the tool definition to the API call with `tool_choice` forcing the function call; (3) receives the tool call arguments JSON; (4) calls `MyModel.model_validate_json()` to parse and validate; (5) if validation fails, optionally retries with the validation error message added to the conversation (auto-correction loop). The result is a fully typed, validated Pydantic object.

**Q5. Why are `Optional[...]` fields important when defining Pydantic models for LLM outputs?**

> LLMs can't always determine every field — if asked to extract `rating_prediction` from a review that doesn't mention stars, the model might hallucinate a value. `Optional[int] = None` tells both Pydantic and the LLM: "this field is not required; if you can't determine it, return null." Pydantic marks it as non-required in the JSON Schema, which the LLM reads as permission to omit it. This prevents hallucination while still capturing the field when present.

### Applied / Code-level

**Q6. Walk through the full pipeline for reliable structured LLM output using instructor and Pydantic.**

```python
# 1. Define the shape of what you want
from pydantic import BaseModel, Field
from typing import List, Optional
import instructor
from openai import OpenAI

class ProductInfo(BaseModel):
    name: str    = Field(description="Product name")
    price: float = Field(description="Price in USD as a number")
    in_stock: bool
    features: List[str] = Field(description="Key product features as bullet points")
    rating: Optional[float] = Field(None, description="Average rating 0-5 if mentioned")

# 2. Wrap your client
client = instructor.from_openai(OpenAI())

# 3. Make the call with response_model
result = client.chat.completions.create(
    model="gpt-4.1",
    response_model=ProductInfo,
    messages=[{"role": "user", "content": "Extract product info: 'Sony WH-1000XM5 headphones, $349, available now, excellent ANC, 30hr battery.'"}]
)

# 4. Use the result — it's typed, validated, and ready
print(result.name)      # "Sony WH-1000XM5"
print(result.price)     # 349.0  (float, not "349")
print(result.in_stock)  # True
print(result.features)  # ['excellent ANC', '30hr battery']
```

**Q7. You receive a `ValidationError` when parsing an LLM response with instructor. How do you debug it?**

> ```python
> from pydantic import ValidationError
> try:
>     result = client.chat.completions.create(..., response_model=MyModel)
> except ValidationError as e:
>     for error in e.errors():
>         print(f"Field: {error['loc']}")     # which field failed
>         print(f"Problem: {error['msg']}")   # why it failed
>         print(f"Value: {error['input']}")   # what the LLM returned
> ```
> Common fixes: (1) If the field is sometimes missing, make it `Optional[...]`. (2) If the LLM returns wrong type, add `Field(description=...)` to clarify expected type. (3) If the LLM returns a non-Enum value for an Enum field, list valid values in the description.

**Q8. How do nested Pydantic models appear in JSON Schema, and why does this matter for tool calling?**

> Nested models appear as `$defs` entries with `$ref` pointers. Example:
> ```json
> {"$defs": {"DateRange": {"properties": {"start": {"type": "string"}, "end": {"type": "string"}}}},
>  "properties": {"daterange": {"$ref": "#/$defs/DateRange"}}}
> ```
> OpenAI's tool calling API fully supports `$defs/$ref` — it reads the nested structure correctly and instructs the model accordingly. This matters because it allows arbitrarily deep nesting: a `SearchQuery` containing a `DateRange` containing nested `datetime.date` fields — all validated recursively by Pydantic.

**Q9. What is the difference between `model_dump()` and `model_dump(mode="json")` for nested models?**

> `model_dump()` returns a Python dict where nested objects remain as their Python types (e.g., `datetime.date` objects). Safe for in-memory use, but NOT JSON-serializable. `model_dump(mode="json")` converts all Python types to JSON-compatible equivalents (e.g., `datetime.date(2024, 1, 1)` → `"2024-01-01"`). Use `mode="json"` when you need to call `json.dumps()` or pass to a REST API.

**Q10. A junior engineer on your team proposes: "Instead of Pydantic and instructor, let's just strip markdown and call `json.loads()` with a try-except." What do you say?**

> This handles Failure 1 (markdown wrapping) but completely misses Failure 2 (type mismatches) and Failure 3 (missing/hallucinated fields). `json.loads()` succeeds on syntactically valid JSON regardless of whether types are correct. You'd need to manually check every field's type — essentially reimplementing Pydantic. Pydantic's `ValidationError` is also informative (tells you which field failed and why), whereas manual checks produce ad hoc error messages. Additionally, `instructor`'s retry loop automatically sends the validation error back to the model for self-correction — a capability you'd have to build from scratch. The complexity doesn't pay off.

---

## Key Learning Thoughts — Section 3

1. **LLM output is untrusted input.** Always treat it like user input from a web form — validate, coerce, and reject clearly. The pipeline is: LLM → `model_validate_json()` → typed Python object → your code. Never let raw LLM strings directly enter your business logic.

2. **`@dataclass` is a data container; `BaseModel` is a data contract.** This distinction matters enormously in LLM systems. A data container stores whatever you give it. A data contract enforces what you said you'd accept. In a world where LLMs hallucinate, you need contracts.

3. **`model_json_schema()` makes your Python model the single source of truth** for both the LLM prompt (what it should return) and your application code (what types it gets). Changes propagate automatically in both directions. This is the right engineering practice.

4. **Instructor is thin but valuable.** It doesn't replace Pydantic; it wires Pydantic to the LLM API and adds retry logic. The retry loop is particularly important: if the LLM returns invalid JSON or the wrong type, instructor sends the validation error back as context and asks the model to try again — often succeeding on the second attempt without human intervention.

5. **Nested models communicate semantics to the LLM.** Naming a nested model `DateRange` with fields `start` and `end` communicates the relationship more clearly than flat `start_date` and `end_date` strings. The LLM reads model structure as hints about intent.

6. **`Optional[...]` fields with `= None` defaults are the hallucination safety valve.** For any field the model might not be able to determine from the input, mark it optional. This prevents the model from making up values to satisfy a required field. Be explicit about when you prefer "I don't know" over a guess.

7. **The big picture:** This section is the answer to "how do I use LLMs reliably in production." The answer is: Pydantic for the data contract, `model_json_schema()` to communicate that contract to the LLM, tool calling or instructor to enforce it, and `ValidationError` to catch any failures at the boundary.

---

## Complete Pipeline Summary

```
User input (natural language)
         ↓
    LLM API call
    (with tool schema from model_json_schema() or instructor response_model)
         ↓
    API forces structured JSON output
         ↓
    Pydantic model_validate_json()
    ├── Type coercion ("25" → 25)
    ├── Enum validation ("positive" ∈ Sentiment)
    ├── Optional field handling (null → None)
    └── ValidationError if invalid (→ retry via instructor)
         ↓
    Typed Python object (fully validated)
         ↓
    Your application code (safe to use)
```

---

## Setup Reference

```bash
pip install pydantic instructor openai groq
```

```python
# Core imports
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from enum import Enum
import instructor
from openai import OpenAI

# Wrap client once at startup
client = instructor.from_openai(OpenAI(api_key="..."))

# Define your model
class MyModel(BaseModel):
    field: str = Field(description="...")

# Call
result = client.chat.completions.create(
    model="gpt-4.1",
    messages=[...],
    response_model=MyModel,
)
# result is a typed, validated MyModel instance
```

## Notebook Reference

| File | Section in Notebook |
|------|---------------------|
| `llm_structured_outputs.ipynb` | Section 2: Pydantic validation + `model_json_schema()` |
| `llm_structured_outputs.ipynb` | Section 3: Tool Calling + instructor (OpenAI + Groq) |
| `llm_structured_outputs.ipynb` | Section 4: Nested models + Semantic Search Query + Review Analyzer |
