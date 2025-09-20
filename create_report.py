import os
import json
from datetime import datetime
from typing import List, Dict
import re

from openai import OpenAI

from env_bootstrap import boot
from utils_logging import setup_logger, print_runtime_diag, utc_now_iso
from market_snapshot import get_snapshot, snapshot_to_md
from news_fetch import fetch_topic, Article, RECENT_DAYS
from report_renderer import render_market_section, render_topic_section, render_briefs

from topics_list import MACRO_KEYWORDS as DEFAULT_TOPICS

# -------- 基础配置 --------
TOPICS = DEFAULT_TOPICS

APPEND_STALE = True
MAX_TOP_SECTIONS = 10


def _date_from_url(url: str):
    m = re.search(r"/(20\d{2})[-/](\d{2})[-/](\d{2})/", url)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _openai_client():
    return OpenAI()  # 读取 .env 中的 OPENAI_API_KEY


# create_report.py（仅替换 _summarize_points）
def _summarize_points(client: OpenAI, model: str, topic: str, articles: List[Article]) -> Dict[str, str]:
    sys_prompt = (
        "You are a macro analyst. You MUST ONLY use facts explicitly present in the items. "
        "Do NOT invent entities/numbers. If a required figure is missing in all items, write 'not stated'. "
        "Prefer concrete numbers with units and dates (e.g., 'cut 25bp to 5.25%-5.50% on 2025-09-18')."
    )

    bullets = []
    for a in articles[:3]:
        d = a.published_at.date().isoformat() if a.published_at else "n/a"
        snippet = (a.text or "")[:600].replace("\n", " ").strip()
        bullets.append(f"- date:{d} | src:{a.source} | title:{a.title} | url:{a.url} | text:{snippet}")

    user_prompt = (
        f"Topic: {topic}\n"
        "From ONLY the list below, extract a concise brief in JSON with EXACT keys: what, so_what, who, watch.\n"
        "- In 'what', include the key quantitative facts if present (e.g., size of move in bp, previous vs new level, effective date).\n"
        "- If numbers/levels are not in the texts, say 'not stated'.\n\n"
        + "\n".join(bullets)
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": user_prompt}],
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        bad = not data or sum(1 for k in ("what", "so_what", "who", "watch") if not data.get(k)) >= 3
        if bad:
            return {
                "what": "Insufficient facts",
                "so_what": "Insufficient facts",
                "who": "Insufficient facts",
                "watch": "Insufficient facts",
            }
        return data
    except Exception:
        return {
            "what": "Insufficient facts",
            "so_what": "Insufficient facts",
            "who": "Insufficient facts",
            "watch": "Insufficient facts",
        }


def main():
    # 环境与日志
    dotenv_path = boot()
    logger = setup_logger()
    print_runtime_diag(dotenv_path)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    logger.info("== Morning Report started ==")
    logger.info(f"UTC now: {utc_now_iso()}  | MODEL: {model}")

    # 市场快照
    snapshot = get_snapshot()
    snapshot_md = snapshot_to_md(snapshot)

    client = _openai_client()
    LLM_USED = False
    appendix_items = []

    bundles = []
    engine_stats = {"topics": 0, "recent_used": 0, "stale_used": 0}

    for topic in TOPICS:
        logger.info(f"Collecting for topic: {topic}")
        buckets = fetch_topic(topic)  # {"recent": [...], "stale":[...]}
        recent = buckets["recent"]
        stale = buckets["stale"]

        engine_stats["topics"] += 1
        engine_stats["recent_used"] += len(recent)
        engine_stats["stale_used"] += len(stale)

        # 按 score 排序（高→低）
        recent_sorted = sorted(recent, key=lambda a: getattr(a, "score", 0), reverse=True)
        topic_score = getattr(recent_sorted[0], "score", 0) if recent_sorted else 0

        bundles.append({
            "topic": topic,
            "recent_sorted": recent_sorted,
            "stale": stale,
            "topic_score": topic_score,
        })

    # --- 统一排序 & 只取 Top-10 ---
    bundles.sort(key=lambda b: b["topic_score"], reverse=True)
    selected = bundles[:MAX_TOP_SECTIONS]

    sections = []
    appendix_items = []

    for b in selected:
        topic = b["topic"]
        recent_sorted = b["recent_sorted"]
        stale = b["stale"]

        # Deep/Short 二选一
        if len(recent_sorted) >= 2:
            top_articles = recent_sorted[:2]
            summary = _summarize_points(client, model, topic, top_articles)
            if "Insufficient" not in "".join(summary.values()):
                LLM_USED = True
                sections.append(
                    render_topic_section(
                        f"{topic} (Deep)",
                        [a.title for a in top_articles],
                        summary,
                    )
                )
            else:
                for a in top_articles:
                    appendix_items.append({
                        "title": a.title,
                        "url": a.url,
                        "date": (a.published_at.date().isoformat() if a.published_at else _date_from_url(a.url)),
                        "label": a.source,
                    })

        elif len(recent_sorted) == 1:
            one = recent_sorted[:1]
            summary = _summarize_points(client, model, topic, one)
            if "Insufficient" not in "".join(summary.values()):
                LLM_USED = True
                sections.append(
                    render_topic_section(
                        f"{topic} (Short)",
                        [a.title for a in one],
                        summary,
                    )
                )
            else:
                a = one[0]
                appendix_items.append({
                    "title": a.title,
                    "url": a.url,
                    "date": (a.published_at.date().isoformat() if a.published_at else _date_from_url(a.url)),
                    "label": a.source,
                })
        else:
            if APPEND_STALE:
                for a in stale[:3]:
                    appendix_items.append({
                        "title": a.title,
                        "url": a.url,
                        "date": (a.published_at.date().isoformat() if a.published_at else _date_from_url(a.url)),
                        "label": a.source,
                    })

    # 其余未入选 Top-10 的主题，也把部分链接丢到 Briefs（可选）
    for b in bundles[MAX_TOP_SECTIONS:]:
        extras = []
        extras.extend(b["recent_sorted"][:2])
        if APPEND_STALE:
            extras.extend(b["stale"][:2])
        for a in extras:
            appendix_items.append({
                "title": a.title,
                "url": a.url,
                "date": (a.published_at.date().isoformat() if a.published_at else _date_from_url(a.url)),
                "label": a.source,
            })

    # Assemble markdown
    md_parts = []
    md_parts.append("# Daily Macro Morning Brief\n")
    md_parts.append(render_market_section(snapshot_md))
    md_parts.extend(sections)
    md_parts.append(render_briefs(appendix_items))

    out_name = f"Macro_Report_{datetime.utcnow().date().isoformat()}.md"
    with open(out_name, "w", encoding="utf-8") as f:
        f.write("\n\n".join(md_parts))

    logger.info(
        f"[SUMMARY] LLM_USED={LLM_USED} | topics={engine_stats['topics']} | "
        f"recent_used={engine_stats['recent_used']} | stale_used={engine_stats['stale_used']}"
    )
    logger.info(f"Report written to: {out_name}")


if __name__ == "__main__":
    main()
