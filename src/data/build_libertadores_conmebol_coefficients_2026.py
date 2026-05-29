"""Build a club rankings CSV from the CONMEBOL 2026 PDF.

The output intentionally mirrors the columns expected by the current club
preprocessor, even though the code still uses UEFA-flavoured field names.
It also appends FIFA country ranking context from ``data/ranking_fifa_2025.csv``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pdfplumber


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF_PATH = REPO_ROOT.parent / "Ranking-de-Clubes-CONMEBOL-2026.pdf"
DEFAULT_FIFA_PATH = REPO_ROOT / "data" / "ranking_fifa_2025.csv"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "preprocessing_data" / "libertadores_conmebol_coefficients_2026.csv"
PDF_SOURCE_URL = "local:Ranking-de-Clubes-CONMEBOL-2026.pdf"

PDF_COUNTRY_TO_ENGLISH = {
    "Argentina": "Argentina",
    "Bolivia": "Bolivia",
    "Brasil": "Brazil",
    "Chile": "Chile",
    "Colombia": "Colombia",
    "Ecuador": "Ecuador",
    "Mexico": "Mexico",
    "Paraguay": "Paraguay",
    "Perú": "Peru",
    "Uruguay": "Uruguay",
    "Venezuela": "Venezuela",
}

TEAM_ALIAS_OVERRIDES: Dict[Tuple[str, str], str] = {
    ("Argentinos Juniors", "Argentina"): "Arg Juniors",
    ("Atlético Tucumán", "Argentina"): "Atlé Tucumán",
    ("Barcelona de Guayaquil", "Ecuador"): "Barcelona",
    ("Botafogo", "Brazil"): "Botafogo (RJ)",
    ("Central Córdoba", "Argentina"): "Cen. Córdoba-SdE",
    ("Club Guaraní", "Paraguay"): "Guaraní",
    ("Defensor Sporting Club", "Uruguay"): "Defensor Sporting",
    ("Deportivo La Guaira", "Venezuela"): "Dep. La Guaira",
    ("Estudiantes de La Plata", "Argentina"): "Estudiantes-LP",
    ("Estudiantes de Mérida", "Venezuela"): "Estudiantes Mérida",
    ("Everton de Viña del Mar", "Chile"): "Everton",
    ("Fortaleza EC", "Brazil"): "Fortaleza",
    ("Independiente del Valle", "Ecuador"): "Independiente",
    ("Independiente Rivadavia", "Argentina"): "Ind. Rivadavia",
    ("Independiente Santa Fe", "Colombia"): "Santa Fe",
    ("Jorge Wilstermann", "Bolivia"): "Wilstermann",
    ("Montevideo City Torque", "Uruguay"): "Torque",
    ("Nacional (Asunción)", "Paraguay"): "Nacional Asunción",
    ("Portuguesa FC (Venezuela)", "Venezuela"): "Portuguesa",
    ("Racing Club (Montevideo)", "Uruguay"): "Racing",
    ("Red Bull Bragantino", "Brazil"): "Bragantino",
    ("Universidad Católica (Ecuador)", "Ecuador"): "Univ Católica",
    ("Universitario de Deportes", "Peru"): "Universitario",
}
UNIQUE_NAME_OVERRIDES: Dict[Tuple[str, str], str] = {
    ("Independiente", "Argentina"): "Independiente (Argentina)",
    ("Libertad", "Ecuador"): "Libertad (Ecuador)",
    ("Portuguesa", "Brazil"): "Portuguesa (Brazil)",
}


def parse_decimal(value: object) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    return float(text.replace(".", "").replace(",", "."))


def load_conmebol_rankings(pdf_path: Path) -> pd.DataFrame:
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if not table:
                    continue
                for row in table:
                    if not row or row[0] == "Posición":
                        continue
                    rows.append(row)

    rankings = pd.DataFrame(
        rows,
        columns=["pdf_rank_raw", "official_name", "country_es", "historical_raw", "performance_raw", "total_raw"],
    )
    rankings = rankings.dropna(subset=["official_name"]).reset_index(drop=True)

    numeric_rank = pd.to_numeric(rankings["pdf_rank_raw"], errors="coerce")
    max_numeric_rank = int(numeric_rank.max())
    sr_mask = numeric_rank.isna()
    rankings["overall_rank"] = numeric_rank
    rankings.loc[sr_mask, "overall_rank"] = range(max_numeric_rank + 1, max_numeric_rank + 1 + int(sr_mask.sum()))
    rankings["overall_rank"] = rankings["overall_rank"].astype(int)

    rankings["historical_club_coefficient"] = rankings["historical_raw"].map(parse_decimal)
    rankings["season_club_coefficient"] = rankings["performance_raw"].map(parse_decimal)
    rankings["overall_club_coefficient"] = rankings["total_raw"].map(parse_decimal)
    rankings["country"] = rankings["country_es"].map(PDF_COUNTRY_TO_ENGLISH).fillna(rankings["country_es"])
    rankings["rank_status"] = rankings["pdf_rank_raw"].where(rankings["pdf_rank_raw"].eq("S/R"), "ranked")

    season_sorted = rankings.sort_values(
        ["season_club_coefficient", "overall_club_coefficient", "overall_rank", "official_name"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    season_sorted["season_rank"] = range(1, len(season_sorted) + 1)
    rankings = rankings.merge(
        season_sorted.loc[:, ["official_name", "country", "season_rank"]],
        on=["official_name", "country"],
        how="left",
    )
    return rankings


def load_fifa_rankings(fifa_path: Path) -> pd.DataFrame:
    fifa = pd.read_csv(fifa_path).copy()
    fifa["ranking"] = pd.to_numeric(fifa["ranking"], errors="coerce")
    fifa["points"] = pd.to_numeric(fifa["points"], errors="coerce")
    return fifa.rename(
        columns={
            "team": "fifa_country_team",
            "ranking": "fifa_country_rank",
            "points": "fifa_country_points",
            "confederation": "fifa_country_confederation",
            "squad_size": "fifa_country_squad_size",
            "avg_age": "fifa_country_avg_age",
            "total_value": "fifa_country_total_value",
        }
    )


def resolve_unique_official_name(official_name: str, country: str) -> str:
    return UNIQUE_NAME_OVERRIDES.get((official_name, country), official_name)


def resolve_team_alias(official_name: str, unique_official_name: str, country: str) -> str:
    override = TEAM_ALIAS_OVERRIDES.get((official_name, country))
    if override:
        return override
    return unique_official_name


def build_output(rankings: pd.DataFrame, fifa: pd.DataFrame) -> pd.DataFrame:
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for _, row in rankings.sort_values(["overall_rank", "official_name"]).iterrows():
        official_name = str(row["official_name"])
        country = str(row["country"])
        unique_official_name = resolve_unique_official_name(official_name, country)
        team_alias = resolve_team_alias(official_name, unique_official_name, country)
        alias_overridden = (official_name, country) in TEAM_ALIAS_OVERRIDES
        rows.append(
            {
                "team": team_alias,
                "display_name": team_alias,
                "official_name": unique_official_name,
                "pdf_official_name": official_name,
                "_unique_official_name": unique_official_name,
                "_alias_overridden": int(alias_overridden),
                "overall_source_team_name": team_alias,
                "overall_source_team_official_name": official_name,
                "overall_rank": int(row["overall_rank"]),
                "overall_club_coefficient": float(row["overall_club_coefficient"]),
                "season_source_team_name": team_alias,
                "season_source_team_official_name": official_name,
                "season_rank": int(row["season_rank"]),
                "season_club_coefficient": float(row["season_club_coefficient"]),
                "country": country,
                "uefa_season_year": 2026,
                "historical_club_coefficient": float(row["historical_club_coefficient"]),
                "conmebol_rank_status": row["rank_status"],
                "pdf_rank_raw": row["pdf_rank_raw"],
                "overall_source_url": PDF_SOURCE_URL,
                "season_source_url": PDF_SOURCE_URL,
                "scraped_at_utc": now,
            }
        )

    out = pd.DataFrame(rows)
    dup_mask = out["team"].duplicated(keep=False)
    if dup_mask.any():
        for team_name, group in out.loc[dup_mask].groupby("team", sort=False):
            keep_idx = None
            overridden = group.loc[group["_alias_overridden"] == 1]
            if len(overridden) == 1:
                keep_idx = overridden.index[0]
            else:
                keep_idx = group.sort_values(["overall_rank", "official_name"]).index[0]

            update_idx = [idx for idx in group.index if idx != keep_idx]
            out.loc[update_idx, "team"] = out.loc[update_idx, "_unique_official_name"]
            out.loc[update_idx, "display_name"] = out.loc[update_idx, "_unique_official_name"]
            out.loc[update_idx, "overall_source_team_name"] = out.loc[update_idx, "_unique_official_name"]
            out.loc[update_idx, "season_source_team_name"] = out.loc[update_idx, "_unique_official_name"]

    out = out.merge(
        fifa[
            [
                "fifa_country_team",
                "fifa_country_rank",
                "fifa_country_points",
                "fifa_country_confederation",
                "fifa_country_squad_size",
                "fifa_country_avg_age",
                "fifa_country_total_value",
            ]
        ],
        left_on="country",
        right_on="fifa_country_team",
        how="left",
    )

    missing_fifa = out.loc[out["fifa_country_rank"].isna(), "country"].unique().tolist()
    if missing_fifa:
        raise ValueError(f"Missing FIFA country rankings for: {missing_fifa}")

    # Keep country-strength features in the same shape expected by the
    # existing club pipeline, using FIFA ranking/points as the country proxy.
    out["overall_country_coefficient"] = out["fifa_country_points"].astype(float)
    out["season_country_coefficient"] = out["fifa_country_points"].astype(float)
    out["country_uefa_overall_rank"] = out["fifa_country_rank"].astype(float)
    out["country_uefa_season_rank"] = out["fifa_country_rank"].astype(float)

    out = out.drop(columns=["_unique_official_name", "_alias_overridden"])
    return out.sort_values(["overall_rank", "team"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument("--fifa", type=Path, default=DEFAULT_FIFA_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    rankings = load_conmebol_rankings(args.pdf)
    fifa = load_fifa_rankings(args.fifa)
    out = build_output(rankings, fifa)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows to {args.out}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
