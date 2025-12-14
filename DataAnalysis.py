import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pymongo
import polars as pl
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib


class DataAnalysis:
    def __init__(self):
        sns.set(font_scale=1.8)
        with pymongo.MongoClient() as c:
            db = c.get_database("eltek")
            self.users = db.get_collection("devices")
            self.water = db.get_collection("water-flow")

    @staticmethod
    def get_N_colors(n, cmap_name="viridis"):
        """
        Returns a list of N distinct colors (RGBA tuples) from a Matplotlib colormap.
        """
        # 1. Get the Colormap object
        cmap = matplotlib.colormaps[cmap_name]

        # 2. Generate N equally spaced values between 0 and 1
        # These values represent points along the colormap
        color_indices = np.linspace(0, 1, n)

        # 3. Call the Colormap object with the indices to get the RGBA colors
        colors = cmap(color_indices)

        return colors.tolist()

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
            {
                "$project": {
                    "_id": 0,
                    "date": "$date",
                    "flow": "$water-flow.value",
                    "water_temp": "$water-temperature.value",
                    "ambient_temp": "$ambient-temperature.value",
                }
            },
        ]
        df = self.water.aggregate(query)
        df = pl.DataFrame(list(df))
        return df

    @staticmethod
    def get_activities(df: pl.DataFrame, threshold=0, label="flow"):
        df = (
            df.with_columns(
                # 1. Create a boolean column: True if flow is above the threshold
                is_active=pl.col(label) > threshold,
                temp_delta=(pl.col("water_temp") - pl.col("ambient_temp")),
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
        df_features = (
            df.group_by("activity_id")
            .agg(
                [
                    # Timing
                    pl.col("date").min().alias("start_time"),
                    pl.col("date").max().alias("end_time"),
                    # Flow characteristics
                    pl.col(label).sum().alias("total_volume"),
                    pl.col(label).max().alias("peak_flow"),
                    pl.col(label).mean().alias("avg_flow"),
                    pl.col(label)
                    .std()
                    .fill_null(0)
                    .alias("flow_stability"),  # 0 std = very stable
                    # Temperature characteristics (Crucial for appliance detection)
                    pl.col("temp_delta").max().alias("max_temp_delta"),
                    pl.col("temp_delta").mean().alias("avg_temp_delta"),
                ]
            )
            .with_columns(
                duration=(pl.col("end_time") - pl.col("start_time")).dt.total_seconds()
            )
            .sort("start_time")
        )
        return df_features

    @staticmethod
    def cluster_activities(activities_df: pl.DataFrame, n_clusters=5):
        """
        Applies K-Means to categorize water events.
        """
        # Select features for clustering
        features = [
            "duration",
            "total_volume",
            "peak_flow",
            # "avg_temp_delta",
            "flow_stability",
        ]

        # Convert to numpy for sklearn
        X = activities_df.select(features).to_numpy()

        # Normalize data (Critical: duration is in seconds, volume in liters - scales differ)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(X)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(x_scaled)

        # Attach labels back to Polars DataFrame
        return activities_df.with_columns(pl.Series("usage_cluster", labels))

    @staticmethod
    def detect_micro_leaks(df: pl.DataFrame, window_minutes=60):
        """
        Detects if the minimum flow never drops to zero within a sliding window.
        """
        # Calculate rolling min over the window
        # Note: '1i' in Polars depends on index, if index is time, use dynamic rolling
        leak_df = (
            df.sort("date")
            .set_sorted("date")
            .with_columns(
                rolling_min_flow=pl.col("flow").rolling_min_by(
                    "date", f"{window_minutes}m"
                )
            )
        )

        # If the rolling minimum is > 0 for a sustained period, you have a leak
        leaks = leak_df.filter(pl.col("rolling_min_flow") > 0.05)  # 0.05 lt/m buffer

        return leaks

    def plot_activities(
            self,
        df: pl.DataFrame,
        activities: pl.DataFrame,
        label="flow",
        duration_th=(1, 500),
        peak_th=1,
        is_cluster=True
    ):
        fig, ax = plt.subplots()
        df.to_pandas().set_index("date", drop=True)[label].plot(ax=ax)

        if is_cluster:
            n = len(activities["usage_cluster"].unique())
            colors = self.get_N_colors(n)
            clusters = activities["usage_cluster"].unique()
            for i, c in enumerate(clusters):
                x = activities.filter(pl.col("usage_cluster").eq(c))
                for row in x.iter_rows(named=True):
                    start, end = row["start_time"], row["end_time"]
                    ax.axvspan(start, end, color=colors[i], alpha=0.3)
        else:
            x = activities.filter(
                (pl.col("duration").lt(duration_th[1]))
                & (pl.col("duration").gt(duration_th[0]))
                & (pl.col("peak_flow").gt(peak_th))
            )
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


    @staticmethod
    def describe_clusters(activities: pl.DataFrame, cluster_col="usage_cluster") -> tuple[pl.DataFrame, dict]:
        """
        Analyzes clusters and assigns human-readable labels based on water usage patterns.
        Returns:
            1. summary_df: Statistical summary of each cluster.
            2. label_map: A dictionary {cluster_id: "Description"}
        """

        # 1. Calculate the "Centroids" (Average behavior per cluster)
        summary = (
            activities.group_by(cluster_col)
            .agg(
                count=pl.len(),
                avg_duration=pl.col("duration").mean(),
                avg_volume=pl.col("total_volume").mean(),
                avg_peak=pl.col("peak_flow").mean(),
                avg_temp_delta=(
                    pl.col("avg_temp_delta").mean()
                    if "avg_temp_delta" in activities.columns else pl.lit(0)
                ),
                avg_flow_stability=(
                    pl.col("flow_stability").mean()
                    if "flow_stability" in activities.columns else pl.lit(0)
                )
            )
            .sort(cluster_col)
        )

        # 2. Heuristic Logic to name the clusters
        # We iterate over the summary because the number of clusters is small (e.g., < 20)
        label_map = {}

        for row in summary.iter_rows(named=True):
            cid = row[cluster_col]
            dur = row["avg_duration"]  # seconds
            vol = row["avg_volume"]  # liters
            temp = row["avg_temp_delta"]  # degrees C diff
            peak = row["avg_peak"]  # liters/min

            # --- Logic Tree ---

            # High Temperature Events
            if temp > 5.0:
                if vol > 25 and dur > 120:
                    desc = "Shower / Bath"
                elif vol > 10:
                    desc = "Hot Water Wash (Dishes/Face)"
                else:
                    desc = "Short Hot Tap Usage"

            # Cold/Ambient Temperature Events
            else:
                if vol > 30 and dur > 300:
                    desc = "Irrigation / Hose Usage"
                elif 4 <= vol <= 12 and peak > 5:
                    # Toilets usually dump 6L-9L very quickly (high peak)
                    desc = "Toilet Flush"
                elif vol > 15 and dur > 600:
                    # Washing machines/Dishwashers often have long durations with low avg temp (if cold fill)
                    # Note: This is tricky as machines pulse water.
                    desc = "Appliance Cycle (Cold Fill)"
                elif vol < 2 and dur < 30:
                    desc = "Quick Tap (Glass of Water/Hands)"
                else:
                    desc = f"General Cold Usage (Vol: {vol:.1f}L)"

            label_map[cid] = desc
        print(label_map)
        # 3. Attach the inferred label back to the summary for easy reading
        summary = summary.with_columns(
            pl.col(cluster_col).cast(pl.String).replace(label_map).alias("inferred_category")
        )

        return summary, label_map

    def run(self):
        df = self.load_water(
            device_id="EWCS30065",
            start=datetime.datetime(2025, 9, 10),
            end=datetime.datetime(2025, 9, 20),
        )
        self.plot_water(df)
        print(df)
        activities = self.get_activities(df, threshold=1)
        self.plot_activities(df, activities, is_cluster=False)
        clustered = self.cluster_activities(activities, 5)
        leaks = self.detect_micro_leaks(df, window_minutes=10)
        summary, label_map = self.describe_clusters(clustered)
        print(summary)
        print(label_map)
        self.plot_activities(df, clustered)


if __name__ == "__main__":
    DataAnalysis().run()
