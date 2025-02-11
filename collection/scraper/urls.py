from dataclasses import dataclass


# modeled off this URL, which is all matches from 2024:
# https://www.hltv.org/results?startDate=2024-01-01&endDate=2024-12-31&content=demo&gameType=CS2&matchType=Lan

@dataclass
class ResultsUrl:
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    content: str = "demo"
    game_type: str = "CS2"
    match_type: str = "Lan"
    offset: int = 0

    def __str__(self):
        return (
            f"https://www.hltv.org/results?"
            f"offset={self.offset}&"
            f"startDate={self.start_date}&"
            f"endDate={self.end_date}&"
            f"content={self.content}&"
            f"gameType={self.game_type}&"
            f"matchType={self.match_type}"
        )
