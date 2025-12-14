import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path

class AMPds:
    def __init__(self):
        self.amp_ds_path = Path("data/dataverse_files")

    def load_water(self):
        df_whole = pl.read_csv(
            self.amp_ds_path / "Water_WHW.csv"
        ).with_columns(
            date=pl.from_epoch("unix_ts")
        ).select(["date", "avg_rate"])
        df_dishwasher = pl.read_csv(
            self.amp_ds_path / "Water_DWW.csv"
        ).with_columns(
            date=pl.col("unix_ts").str.replace("+", "", literal=True)
        ).with_columns(
            date=pl.from_epoch(pl.col("date")),
            dishwasher_rate=pl.col("avg_rate"),
        ).select(["date", "dishwasher_rate"])
        df_hot_water = pl.read_csv(
            self.amp_ds_path / "Water_HTW.csv"
        )

        df = df_whole.join(df_dishwasher, on="date")
        print(df)

        df.to_pandas().set_index("date", drop=True).plot(subplots=True)
        plt.show()


if __name__ == '__main__':
    AMPds = AMPds()
    AMPds.load_water()