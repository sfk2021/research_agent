"""Research Desk: Streamlit front end for the LangGraph research agent.

Local:  streamlit run app.py
Cloud:  add GOOGLE_API_KEY and TAVILY_API_KEY in the app's Secrets.
"""
import json
import os
import re

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Research Desk", page_icon="📈", layout="centered")

# On Streamlit Cloud the keys live in st.secrets; agent.py reads them from the environment.
for _k in ("GOOGLE_API_KEY", "TAVILY_API_KEY"):
    try:
        if _k in st.secrets:
            os.environ[_k] = st.secrets[_k]
    except Exception:
        pass  # no secrets file locally; agent.py loads .env instead

from agent import graph, _text  # noqa: E402

MAX_RUNS_PER_SESSION = 5  # protects your API quota on a public portfolio
EXAMPLES = ["Top 5 largest economies by GDP", "Most streamed songs of all time", "Global EV sales by year"]


def extract_json(text: str):
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    a, b = cleaned.find("{"), cleaned.rfind("}")
    if a == -1 or b == -1:
        return None
    try:
        return json.loads(cleaned[a : b + 1])
    except json.JSONDecodeError:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def research(topic: str):
    result = graph.invoke({"messages": [("user", topic)]})
    raw = _text(result["messages"][-1].content)
    return extract_json(raw), raw


def show(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:  # older Streamlit
        st.altair_chart(chart, use_container_width=True)


def draw(c: dict):
    rows = [d for d in c.get("chartData", []) if isinstance(d, dict) and isinstance(d.get("value"), (int, float))]
    if not rows:
        return
    df = pd.DataFrame(rows)
    kind = str(c.get("type", "bar")).lower()
    tip = ["label:N", "value:Q"]
    if kind in ("pie", "doughnut"):
        ch = alt.Chart(df).mark_arc(innerRadius=60 if kind == "doughnut" else 0).encode(
            theta="value:Q", color=alt.Color("label:N", legend=alt.Legend(title=None)), tooltip=tip)
    else:
        x = alt.X("label:N", sort=None, title=c.get("xAxisLabel"), axis=alt.Axis(labelAngle=-30))
        y = alt.Y("value:Q", title=c.get("yAxisLabel"))
        base = alt.Chart(df)
        ch = (base.mark_line(point=True) if kind == "line" else base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4))
        ch = ch.encode(x=x, y=y, tooltip=tip)
    show(ch.properties(height=320))


st.title("Ask a question. Get the numbers charted.")
st.caption("Two LangGraph agents search the web with Tavily, then Gemini turns the figures into charts.")

st.session_state.setdefault("runs", 0)
st.text_input("Research topic", key="q", placeholder="Top 5 largest economies by GDP")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    col.button(ex, on_click=lambda e=ex: st.session_state.update(q=e), use_container_width=True)
go = st.button("Research", type="primary")

topic = st.session_state.q.strip()
if go and topic:
    if st.session_state.runs >= MAX_RUNS_PER_SESSION:
        st.warning("You’ve reached the demo limit for this session. Reload the page to start a new one.")
        st.stop()
    try:
        with st.spinner("Searching the web and building charts. This can take up to a minute…"):
            data, raw = research(topic)
        st.session_state.runs += 1
    except Exception as e:
        st.error(f"The research run failed: {str(e)[:300]}")
        st.stop()

    charts = data.get("chartsArray", []) if isinstance(data, dict) else []
    if not charts:
        st.info("No charts were produced. Try a more specific, measurable question.")
        st.write((data or {}).get("otherResearchFindings") or raw)
        st.stop()

    for c in charts:
        st.subheader(c.get("title", "Chart"))
        draw(c)
        if c.get("chartFootNote"):
            st.caption(c["chartFootNote"])
    if data.get("otherResearchFindings"):
        with st.expander("Other findings"):
            st.write(data["otherResearchFindings"])
    st.download_button("Download chart JSON", json.dumps(data, indent=2), "research.json", "application/json")

st.caption("Figures come from live web searches and can be wrong. Check the sources before relying on them.")
