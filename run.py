import json
from agent import graph

result = graph.invoke({
    "messages": [("user", "Research the top 5 largest economies by GDP and chart them")]
})

content = result["messages"][-1].content
if isinstance(content, list):  # newer Gemini wrappers can return content blocks
    content = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)

content = content.strip().removeprefix("```json").removesuffix("```").strip()
print(content)

try:
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(content), f, indent=2)
    print("\nSaved to output.json")
except json.JSONDecodeError:
    print("\n(Output wasn't valid JSON; see above)")