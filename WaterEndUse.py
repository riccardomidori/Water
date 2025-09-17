from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


class WEU:
    def __init__(self):
        self.dataset_path = Path("data/WEUSEDTO-Data/Dataset")
        self.separators = {
            "bidet": ",",
            "dishwasher": " ",
            "toilet": " ",
            "kitchen": ",",
            "shower": " ",
            "basin": " ",
            "washing-machine": " ",
            "main": " ",
        }
        self.names = {
            "toilet": "feed.Toilet",
            "bidet": "feed_Bidet.MYD",
            "dishwasher": "feed_Dishwasher30.MYD",
            "kitchen": "feed_Kitchenfaucet.MYD",
            "shower": "feed_Shower.MYD",
            "basin": "feed_Washbasin.MYD",
            "washing-machine": "feed_Washingmachine.MYD",
            "main": "feed_WholeHouse.MYD",
        }
        self.headers = {
            "toilet": True,
            "bidet": True,
            "dishwasher": True,
            "kitchen": True,
            "shower": False,
            "basin": False,
            "washing-machine": False,
            "main": False,
        }

    def analysis(self):
        for name, path in self.names.items():
            print(path)
            sep = self.separators[name]
            header = self.headers[name]
            df = (
                pl.read_csv(
                    self.dataset_path / f"{path}.csv", separator=sep, has_header=header
                )
                .rename({"column_1": "Time", "column_2": "Flow"}, strict=False)
                .with_columns(Time=pl.from_epoch("Time"))
            )
            if df.shape[1] > 2:
                df = df.with_columns(EndTime=pl.from_epoch("End time"))
            print(df)
            df.to_pandas().set_index("Time", drop=True)["Flow"].plot()
            plt.show()


if __name__ == "__main__":
    WEU().analysis()
