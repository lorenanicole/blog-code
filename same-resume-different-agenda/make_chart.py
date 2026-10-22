#!/usr/bin/env python3
"""
make_chart.py — Regenerate the demo agenda-gap chart using plotnine.

Reads runs/demo/summary.json and writes:
  ../../content/images/same-resume-different-agenda-audit.png

Usage:
    python make_chart.py
"""

import json
from pathlib import Path
import pandas as pd
from plotnine import (
    ggplot, aes, geom_point, geom_segment, geom_vline,
    scale_color_manual, scale_y_discrete, coord_flip,
    labs, theme, theme_minimal, element_text, element_line,
    element_blank, element_rect, facet_wrap
)

SUMMARY = Path(__file__).parent / "runs/demo/summary.json"
OUT = Path(__file__).parent / "../../content/images/same-resume-different-agenda-audit.png"

BG      = "#0f1117"
AMBER   = "#F59E0B"
TEAL    = "#2DD4BF"
GREY    = "#6B7280"
WHITE   = "#F1F5F9"
DIM     = "#94A3B8"

TERM_LABELS = {
    "remote_hybrid":   "Remote / hybrid work",
    "flexible_hours":  "Flexible hours",
    "reduced_hours":   "Reduced hours",
    "parental_leave":  "Parental leave",
    "childcare":       "Childcare support",
    "wellness":        "Wellness / mental health",
    "pto":             "Extra PTO",
    "health_benefits": "Health benefits",
    "retirement":      "Retirement / 401(k)",
    "bonus":           "Signing / perf bonus",
    "equity":          "Equity / RSUs",
    "raises":          "Raises / salary review",
    "title_promotion": "Title / promotion path",
    "visibility":      "High-visibility projects",
    "relocation":      "Relocation",
    "severance":       "Severance",
    "learning":        "Learning & dev budget",
    "equipment":       "Equipment / tech",
}

def load_gaps(summary: dict, condition: str) -> dict[str, float]:
    return {
        k: v["gap_pp"]
        for k, v in summary["conditions"][condition]["terms"].items()
    }

def main():
    data = json.loads(SUMMARY.read_text())

    rows = []
    for cond, label, color in [
        ("baseline",       "Baseline",        AMBER),
        ("ask_dont_infer", "Ask, don't infer", TEAL),
    ]:
        gaps = load_gaps(data, cond)
        for slug, gap in gaps.items():
            rows.append({
                "term":      TERM_LABELS.get(slug, slug),
                "gap":       gap,
                "condition": label,
                "color":     color,
            })

    df = pd.DataFrame(rows)

    # order terms by baseline gap
    baseline_order = (
        df[df["condition"] == "Baseline"]
        .sort_values("gap")["term"]
        .tolist()
    )
    df["term"] = pd.Categorical(df["term"], categories=baseline_order, ordered=True)

    # segment df for the connector lines
    base = df[df["condition"] == "Baseline"][["term", "gap"]].rename(columns={"gap": "gap_base"})
    fix  = df[df["condition"] == "Ask, don't infer"][["term", "gap"]].rename(columns={"gap": "gap_fix"})
    segs = base.merge(fix, on="term")

    p = (
        ggplot(df, aes(x="term", y="gap", color="condition"))
        + geom_segment(
            aes(x="term", xend="term", y="gap_base", yend="gap_fix"),
            data=segs, color=GREY, size=0.8, inherit_aes=False
          )
        + geom_point(size=3.5, alpha=0.95)
        + geom_vline(xintercept=0, color=DIM, linetype="dashed", size=0.5)
        + scale_color_manual(values={"Baseline": AMBER, "Ask, don't infer": TEAL})
        + coord_flip()
        + labs(
            title="Which terms AI advice recommends to women vs. men",
            subtitle="SYNTHETIC DEMO DATA · gap in recommendation rate (women minus men, pp)",
            x="", y="← more for men   ·   percentage points   ·   more for women →",
            color="",
            caption="Synthetic generator; planted bias. Says nothing about any real model.",
          )
        + theme_minimal(base_family="monospace")
        + theme(
            figure_size=(11, 7),
            plot_background=element_rect(fill=BG),
            panel_background=element_rect(fill=BG),
            panel_grid_major_x=element_line(color="#1e2330", size=0.5),
            panel_grid_major_y=element_blank(),
            panel_grid_minor=element_blank(),
            axis_text=element_text(color=DIM, size=8),
            axis_title_x=element_text(color=DIM, size=8),
            plot_title=element_text(color=WHITE, size=12, weight="bold"),
            plot_subtitle=element_text(color=AMBER, size=8),
            plot_caption=element_text(color=GREY, size=7),
            legend_background=element_rect(fill=BG),
            legend_text=element_text(color=DIM, size=9),
            legend_position="top",
          )
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    p.save(str(OUT), dpi=160, bbox_inches="tight", verbose=False)
    print(f"Saved → {OUT.resolve()}")

if __name__ == "__main__":
    main()
