import pandas as pd

from src.utils.confederation_mapping import add_confederation_to_matches


def test_add_confederation_to_matches(tmp_path):
    confed_path = tmp_path / "confed.csv"
    pd.DataFrame(
        [
            {"country": "Brazil", "confederation": "CONMEBOL"},
            {"country": "Japan", "confederation": "AFC"},
        ]
    ).to_csv(confed_path, index=False)

    matches = pd.DataFrame(
        [
            {"home_team": "Brazil", "away_team": "Japan"},
        ]
    )

    updated = add_confederation_to_matches(matches, confed_path=str(confed_path))

    assert updated.loc[0, "home_team_confederation"] == "CONMEBOL"
    assert updated.loc[0, "away_team_confederation"] == "AFC"
