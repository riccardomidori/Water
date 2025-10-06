import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pymongo
import polars as pl


class DataAnalysis:
    def __init__(self):
        sns.set(font_scale=1.8)
        with pymongo.MongoClient() as c:
            db = c.get_database("eltek")
            self.users = db.get_collection("devices")
            self.water = db.get_collection("water-flow")

    @staticmethod
    def plot_water(df: pl.DataFrame, label="flow"):
        fig, ax = plt.subplots()
        df.to_pandas().set_index("date", drop=True)[label].plot(ax=ax)
        ax.set_title(label)
        plt.show()

    def load_water(
        self,
        device_id="EWCS30065",
        start=datetime.datetime(2025, 9, 1),
        end=datetime.datetime(2025, 9, 7),
    ):
        query = [
            {"$match": {"device_id": device_id, "date": {"$gte": start, "$lt": end}}},
            {"$project": {"_id": 0, "date": "$date", "flow": "$water-flow.value"}},
        ]
        df = self.water.aggregate(query)
        df = pl.DataFrame(list(df))
        df.to_pandas().set_index("date", drop=True).plot(subplots=True)
        plt.show()
        return df

    @staticmethod
    def get_activities(df: pl.DataFrame, threshold=0, label="flow"):
        df = (
            df.with_columns(
                # 1. Create a boolean column: True if flow is above the threshold
                is_active=pl.col(label)
                > threshold
            )
            .with_columns(
                # 2. Identify start of activities: True if `is_active` is True and the previous value was False
                is_start=pl.col("is_active")
                & (~pl.col("is_active").shift(1).fill_null(False))
            )
            .with_columns(
                # 3. Create a group identifier for each activity using a cumulative sum on the `is_start` column
                activity_id=pl.col("is_start").cum_sum()
            )
            .filter(
                # Keep only the rows that are part of an activity
                pl.col("is_active")
            )
        )
        df_activity = (
            df.group_by("activity_id")
            .agg(
                # 4. Aggregate to find the start and end time of each activity
                start_time=pl.col("date").min().dt.offset_by(by="-1s"),
                end_time=pl.col("date").max().dt.offset_by("1s"),
                peak=pl.col("flow").max(),
                consumption=pl.col("flow").sum(),
            )
            .with_columns(
                duration=(pl.col("end_time") - pl.col("start_time")).dt.total_seconds()
            )
            .drop("activity_id")  # Clean up the output by dropping the helper column
            .sort("start_time")
        )  # Sort the results for a clean list)
        return df_activity

    @staticmethod
    def cluster_activities(
        activities: pl.DataFrame, feature1="consumption", feature2="peak"
    ):
        fig, ax = plt.subplots()
        activities.to_pandas()[[feature1, feature2]].plot(
            ax=ax, kind="scatter", x=feature1, y=feature2
        )
        plt.show()

    @staticmethod
    def plot_activities(
        df: pl.DataFrame,
        activities: pl.DataFrame,
        label="flow",
        duration_th=(1, 5),
        peak_th=1,
    ):
        x = activities.filter(
            (pl.col("duration").lt(duration_th[1]))
            & (pl.col("duration").gt(duration_th[0]))
            & (pl.col("peak").gt(peak_th))
        )
        fig, ax = plt.subplots()
        df.to_pandas().set_index("date", drop=True)[label].plot(ax=ax)
        for row in x.iter_rows(named=True):
            start, end = row["start_time"], row["end_time"]
            ax.axvspan(start, end, color=(1, 0, 0, 0.3))
        plt.show()

    @staticmethod
    def analysis_activities(
        df: pl.DataFrame,
        activities: pl.DataFrame,
        duration_th=(1, 5),
        peak_th=1,
        show=False,
    ) -> dict:
        to_filter = (
            (pl.col("duration").lt(duration_th[1]))
            & (pl.col("duration").gt(duration_th[0]))
            & (pl.col("peak").gt(peak_th))
        )
        x = activities.filter(to_filter).with_columns(
            hour_day=pl.col("start_time").dt.hour()
        )
        x_hour = x.group_by("hour_day").agg(count=pl.len()).sort(by="hour_day")

        out = {
            "average_consumption": x["consumption"].mean(),
            "percentage_amount": len(x) / len(activities) * 100,
            # "time_distribution": x_hour,
        }
        print(out)
        if show:
            fig, ax = plt.subplots()
            x_hat = activities.filter(~to_filter)
            mask = df.__deepcopy__()
            for row in x_hat.iter_rows(named=True):
                start, end = row["start_time"], row["end_time"]
                mask = mask.with_columns(
                    flow=pl.when((pl.col("date").gt(start)) & (pl.col("date").lt(end)))
                    .then(0)
                    .otherwise(pl.col("flow"))
                )
            mask.to_pandas().set_index("date", drop=True)["flow"].plot(ax=ax)

            fig, ax = plt.subplots()
            ax.bar(x_hour["hour_day"].to_list(), x_hour["count"].to_list())
            plt.show()
        return out

    def run(self):
        df = self.load_water(
            device_id="EWCS30143",
            start=datetime.datetime(2025, 9, 1), end=datetime.datetime.now()
        )
        activities = self.get_activities(df, threshold=1)
        out = {}
        durations = [
            (0, 5),
            (5, 10),
            (10, 15),
            (15, 20),
            (20, 500),
            (500, 1000),
        ]
        for duration_th in durations:
            obj = self.analysis_activities(
                df, activities, duration_th=duration_th, show=True, peak_th=0
            )
            out[duration_th] = obj
        print(out)


if __name__ == "__main__":
    DataAnalysis().run()
