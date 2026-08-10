import json
from pathlib import Path

path = Path(
    r"C:\Users\Kreig\.cursor\projects\b-Kutunika-programmist-yakutia-profile-climate"
    r"\agent-transcripts\b4c22260-2c8b-46fe-a218-441372a99ce0"
    r"\b4c22260-2c8b-46fe-a218-441372a99ce0.jsonl"
)
out = Path(
    r"C:\Users\Kreig\.cursor\projects\b-Kutunika-programmist-yakutia-profile-climate"
    r"\agent-tools\sverka_extract.txt"
)
out.parent.mkdir(parents=True, exist_ok=True)
keys = ("874", "866", "1366", "844", "783", "736", "612", "797.5", "линейн")
with path.open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as w:
    for i, line in enumerate(f):
        if not any(k in line for k in keys):
            continue
        obj = json.loads(line)
        text = ""
        content = obj.get("message", {}).get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += part.get("text", "")
        elif isinstance(content, str):
            text = content
        if text and any(k in text for k in keys):
            w.write(f"--- line {i} role {obj.get('role')}\n")
            w.write(text[:12000])
            w.write("\n\n")
print("wrote", out)
