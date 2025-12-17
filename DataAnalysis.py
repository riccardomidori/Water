import datetime
from typing import Tuple, Dict, Any

import seaborn as sns
import matplotlib.pyplot as plt
import pymongo
import polars as pl
import numpy as np
from polars import DataFrame
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
    def plot_actual_vs_benchmark(summary_df):
        """
        Compares observed cluster frequency vs Italian Averages.
        """
        # 1. Prepare Data
        # Calculate percentage of total events
        total_events = summary_df["count"].sum()

        df_plot = summary_df.select(
            [
                pl.col("inferred_category").alias("Category"),
                (pl.col("count") / total_events * 100).alias("Actual (%)"),
            ]
        ).to_pandas()

        # 2. Define Italian Benchmarks (Approximate)
        # We map your detailed labels to generic categories
        benchmarks = {
            "Short Usage": 65,  # Taps/Hands
            "Toilet": 18,
            "Appliance": 14,  # Washing Machine pulses
            "Shower": 2,
            "Irrigation": 0.5,
            "Leak": 0,
        }

        # 3. Map your categories to benchmark keys for comparison
        # (Simple string matching logic)
        df_plot["Benchmark (%)"] = df_plot["Category"].apply(
            lambda x: next((v for k, v in benchmarks.items() if k in x), 0)
        )

        # 4. Melt for Side-by-Side Bar Chart
        df_melted = df_plot.melt(
            id_vars="Category",
            value_vars=["Actual (%)", "Benchmark (%)"],
            var_name="Type",
            value_name="Percentage",
        )

        # 5. Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df_melted, x="Category", y="Percentage", hue="Type", palette="muted"
        )

        plt.title("House Usage vs. Italian Average (Event Frequency)")
        plt.ylabel("Frequency (% of Total Events)")
        plt.xlabel("Usage Category")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_cluster_physics(summary_df):
        """
        Plots Clusters based on Volume vs Duration, sized by Frequency.
        Expects summary_df to have columns:
        ['inferred_category', 'avg_duration', 'avg_volume', 'count']
        """
        # Convert to Pandas for plotting
        data = summary_df.to_pandas()

        plt.figure(figsize=(12, 8))

        # Create Scatter Plot
        # Size range (s) scales the bubbles to be visible
        sns.scatterplot(
            data=data,
            x="avg_duration",
            y="avg_volume",
            size="count",
            hue="inferred_category",
            sizes=(100, 2000),  # Min and Max bubble size
            alpha=0.7,
            palette="viridis",
        )

        # Add Reference Zones (The "Physics" Boxes)
        # 1. Toilet Zone (Low Duration, Fixed Volume)
        plt.axhspan(
            4, 12, xmin=0, xmax=0.2, color="red", alpha=0.1, label="Toilet Zone"
        )

        # 2. Shower Zone (High Duration, High Volume)
        plt.axhspan(
            30, 150, xmin=0.3, xmax=1.0, color="blue", alpha=0.1, label="Shower Zone"
        )

        # Log Scale is often better because Showers (100L) are huge compared to Taps (0.5L)
        plt.xscale("log")
        plt.yscale("log")

        plt.title("Cluster Physics: Volume vs Duration (Size = Frequency)")
        plt.xlabel("Average Duration (Seconds) - Log Scale")
        plt.ylabel("Average Volume (Liters) - Log Scale")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

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
                    pl.col("date").min().alias("start_time"),
                    pl.col("date").max().alias("end_time"),
                    pl.col("flow").sum().alias("total_volume"),
                    pl.col("flow").max().alias("peak_flow"),
                    pl.col("flow").mean().alias("avg_flow"),
                    pl.col(label)
                    .std()
                    .fill_null(0)
                    .alias("flow_stability"),  # 0 std = very stable
                    # Temperature characteristics (Crucial for appliance detection)
                    # 1. Flow Stability (Standard Deviation / Mean) = Coefficient of Variation
                    # Low (< 0.1) = Machine/Shower. High (> 0.5) = Hand Usage.
                    (pl.col("flow").std() / pl.col("flow").mean())
                    .fill_null(0)
                    .alias("flow_cv"),
                    # 2. Ramp Up (How fast does it hit 80% of peak?)
                    # This is a simplification. For true slope, we'd need regression,
                    # but 'Time to Peak' is a good proxy.
                    (
                        pl.col("date")
                        .filter(pl.col("flow") >= (pl.col("flow").max() * 0.8))
                        .min()
                        - pl.col("date").min()
                    )
                    .dt.total_seconds()
                    .alias("time_to_peak"),
                    # 3. Time Context (Crucial for Flow-Only)
                    pl.col("date").min().dt.hour().alias("hour_of_day"),
                ]
            )
            .with_columns(
                duration=(pl.col("end_time") - pl.col("start_time")).dt.total_seconds()
            )
            # Create a "Squareness" metric: Avg Flow / Peak Flow
            # 1.0 = Perfect Square (Machine). 0.5 = Triangle (Tap).
            .with_columns(shape_factor=(pl.col("avg_flow") / pl.col("peak_flow")))
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
            "shape_factor",
            # "flow_stability",
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
        is_cluster=True,
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

    def describe_flow_clusters(
        self, activities: pl.DataFrame, cluster_col="usage_cluster"
    ) -> tuple[DataFrame, dict[Any, str]]:
        """

                Cluster Name,Frequency (% of Events),Avg Events / Day,Description (Italian Context)
                Short Usage (Taps),60% - 75%,40 - 60,"Hand washing, brushing teeth, cooking, Bidet usage (specific to Italy, adds ~4-6 events/day)."
                Toilet Flush,15% - 20%,12 - 15,~4-5 flushes per person/day.
                Appliance Pulse,10% - 15%,10 - 20,"Warning: A single Washing Machine cycle creates ~15 distinct ""pulse"" events. Don't confuse these with Taps."
                Shower / Bath,1% - 3%,2 - 3,Typically 0.7 showers per person/day.
                Irrigation / Other,< 1%,0 - 1,Highly seasonal (Summer only).



                Cluster Name,Volume (% of Total Liters),Typical Italian Benchmark
                Shower / Bath,35% - 40%,40L - 60L per event (Italy has smaller tanks than US).
                Toilet Flush,25% - 30%,6L - 9L (Old systems) or 3L/6L (New dual flush).
                Washing Machine,10% - 15%,45L - 60L per cycle (Eco-modes are common).
                Kitchen/Bath Taps,10% - 15%,Short bursts of 0.5L - 2L.
                Dishwasher,4% - 6%,10L - 14L per cycle (Very efficient).
                Leaks,Variable,"In older Italian houses, 5-10% leakage is common."


                Cluster Label,Avg Peak Flow (L/m),Max Peak Flow (L/m),Most Common Hour
                Short Tap Usage,4.6 L/m,14.1 L/m,20:00 (8 PM)
                Appliance (Machine),6.7 L/m,14.1 L/m,07:00 (7 AM)
                Toilet Flush,12.1 L/m,22.8 L/m,07:00 (7 AM)

        """
        summary = (
            activities.group_by(cluster_col)
            .agg(
                avg_duration=pl.col("duration").mean(),
                calc_volume=(pl.col("avg_flow").mean() * pl.col("duration").mean())
                / 60,
                avg_volume=pl.col("total_volume").mean(),
                avg_shape=pl.col("shape_factor").mean(),  # 1=Square, 0.5=Spiky
                avg_cv=pl.col("flow_cv").mean(),  # 0=Stable, 1=Chaotic
                avg_peak=pl.col("peak_flow").mean(),
                most_common_hour=pl.col("hour_of_day").mode().first(),
                presence=pl.len() / len(activities),
                count=pl.len(),
            )
            .sort(cluster_col)
        )

        label_map = {}

        for row in summary.iter_rows(named=True):
            cid = row[cluster_col]
            dur = row["avg_duration"]
            vol = row["avg_volume"]
            shape = row["avg_shape"]  # Closer to 1 means "Square wave" (Machine)
            cv = row["avg_cv"]  # Closer to 0 means "Stable"
            hour = row["most_common_hour"]

            # --- Logic Tree (Flow Only) ---

            vol = row["calc_volume"]  # Recalculated Volume (Avg Flow * Duration / 60)
            shape = row["avg_shape"]  # 1.0 = Square, 0.5 = Triangle

            # 1. Shower / Bath (High Volume)
            if vol > 10:
                desc = "Shower / Bath"

            # 2. Appliance (Machine Pulse)
            # Appliances have very "Square" flow (Solenoid valve)
            # Cluster 1 in your data was perfect here (Shape 0.92, Vol 4.2L)
            elif shape > 0.85 and vol > 2.0:
                desc = "Appliance (Machine)"

            # 3. Toilet Flush (Mechanical Decay)
            # Your data showed large 9-10L flushes with a non-square shape (0.57)
            # This matches the "Tank Refill" curve.
            elif 4.5 <= vol <= 12 and shape < 0.8:
                desc = "Toilet Flush"

            # 4. Short Taps (Dominant Category)
            # Includes Hands, Face, Bidet, Kitchen Rinsing
            elif vol < 4.5:
                desc = "Short Tap Usage"

            else:
                desc = "Generic Usage"

            label_map[cid] = desc
        summary = summary.with_columns(
            pl.col(cluster_col)
            .cast(pl.String)
            .replace(label_map)
            .alias("inferred_category")
        )
        self.plot_cluster_physics(summary)
        self.plot_actual_vs_benchmark(summary)

        return summary, label_map

    @staticmethod
    def describe_clusters(
        activities: pl.DataFrame, cluster_col="usage_cluster"
    ) -> tuple[pl.DataFrame, dict]:
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
                    if "avg_temp_delta" in activities.columns
                    else pl.lit(0)
                ),
                avg_flow_stability=(
                    pl.col("flow_stability").mean()
                    if "flow_stability" in activities.columns
                    else pl.lit(0)
                ),
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
            pl.col(cluster_col)
            .cast(pl.String)
            .replace(label_map)
            .alias("inferred_category")
        )

        return summary, label_map

    def load_users(self):
        return self.users.find({}, {"_id": 0})

    def run(self):
        for c in self.load_users():
            device = c["device_id"]
            df = self.load_water(
                device_id=device,
                start=datetime.datetime(2025, 10, 1),
                end=datetime.datetime(2025, 10, 30),
            )
            # self.plot_water(df)
            print(df)
            activities = self.get_activities(df, threshold=1)
            print(activities)
            # activities.write_csv(f"data/{device}_activities.csv")
            # self.plot_activities(df, activities, is_cluster=False)
            clustered = self.cluster_activities(activities, 6)
            summary, label_map = self.describe_flow_clusters(clustered)
            print(summary)
            print(label_map)
            self.plot_activities(df, clustered)


if __name__ == "__main__":
    DataAnalysis().run()
