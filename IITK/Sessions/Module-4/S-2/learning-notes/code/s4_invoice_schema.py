from pathlib import Path
ASSETS = Path(__file__).resolve().parent.parent / "assets"
HERE = Path(__file__).resolve().parent
import json, datetime
from PIL import Image
from pydantic import BaseModel, ValidationError
from vlm_small import ask

class Invoice(BaseModel):          # the contract your downstream code relies on
    vendor: str
    invoice_number: str
    date: datetime.date
    total: float

img = Image.open(HERE / "invoice.png")
prompt = ('Extract vendor, invoice number, date, and total from this invoice. '
          'Respond ONLY with one flat JSON object with exactly these keys: '
          '{"vendor": string, "invoice_number": string, "date": "YYYY-MM-DD", "total": number}')
raw = ask(img, prompt, max_new_tokens=200)
print("RAW:", raw)
try:
    inv = Invoice.model_validate_json(raw[raw.index("{"): raw.rindex("}") + 1])
    print("VALID:", inv)
except (ValueError, ValidationError) as e:
    print("REJECTED ->"); [print("  ", err["loc"], "|", err["msg"], "| got:", repr(err["input"])) for err in e.errors()]
