import datetime
from dataclasses import dataclass
import numpy as np
import polars as pl
import pymongo
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans


@dataclass
class WaterSensorConfig:
    # Limiti fisici di sanificazione sensore (basati sul tuo describe: p50=4345, max=1e6)
    min_valid_pressure_mbar: float = 500.0  # Sotto 0.5 bar è disconnessione/svuotamento impianto
    max_valid_pressure_mbar: float = 12000.0  # Sopra 12 bar è glitch del sensore (es. 1e6)
    max_valid_flow_lpm: float = 60.0  # Sopra 60 l/m in casa è errore di lettura

    # Parametri estrazione eventi
    fallback_flow_threshold_lpm: float = 0.05  # Usato se l'auto-calibrazione non ha abbastanza dati
    max_sampling_gap_sec: int = 10  # Gap massimo per integrazione volume (1 Hz)
    merge_short_pause_sec: int = 12  # Unisce micro-pause < 12s (es. lavabo/doccia)
    appliance_window_sec: int = 2700  # Finestra 45 min per treni d'impulsi (lavatrice/lavastoviglie)

    # Soglie Leak & Anomaly Detection
    mnf_window_hours: tuple[int, int] = (2, 5)  # Finestra notturna 02:00 - 05:00
    burst_duration_sec: int = 2400  # > 40 min flusso ininterrotto
    water_hammer_spike_mbar: float = 800.0  # Sovrapressione post-chiusura rispetto a statica (~4345 mbar)
    isolation_forest_contamination: float = 0.03  # Percentuale attesa di eventi anomali (3%)


class WaterAnalyticsService:
    def __init__(
            self,
            mongo_uri: str = "mongodb://localhost:27017/",
            config: WaterSensorConfig = WaterSensorConfig(),
    ):
        self.mongo_uri = mongo_uri
        self.cfg = config
        sns.set_theme(style="whitegrid", font_scale=1.2)

    def load_water(
            self,
            device_id: str = "EWCS30065",
            start: datetime.datetime = datetime.datetime(2025, 9, 1),
            end: datetime.datetime = datetime.datetime(2026, 9, 7),
    ) -> pl.DataFrame:
        """Estrae i dati da MongoDB ed esegue il pre-processing fisico e la rimozione di glitch (1e6)."""
        query = [
            {"$match": {"device_id": device_id, "date": {"$gte": start, "$lt": end}}},
            {
                "$project": {
                    "_id": 0,
                    "date": "$date",
                    "flow": "$water-flow.value",
                    "water_temp": "$water-temperature.value",
                    "ambient_temp": "$ambient-temperature.value",
                    "water_pressure": "$water-pressure.value",
                }
            },
            {"$sort": {"date": 1}},
        ]
        with pymongo.MongoClient(self.mongo_uri) as client:
            db = client.get_database("eltek")
            cursor = db.get_collection("water-flow").aggregate(query, allowDiskUse=True)
            raw_df = pl.DataFrame(list(cursor))

        if raw_df.is_empty():
            raise ValueError(f"Nessun dato trovato per device_id={device_id} nell'intervallo specificato.")

        return self.preprocess_and_sanitize(raw_df)

    def preprocess_and_sanitize(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        1. Rimuove gli outlier hardware (come water_pressure = 1e6) sostituendoli per interpolazione/forward-fill.
        2. Calcola il cut-off minimo del flussimetro di alta qualità.
        3. Calcola i volumi reali in litri (flow * dt / 60) e la caduta di pressione rispetto alla baseline (~4345 mbar).
        """
        # Auto-calibrazione della soglia di zero del sensore di flusso
        positive_flows = df.filter(pl.col("flow") > 0.0)["flow"]
        if len(positive_flows) > 100:
            detected_min_flow = float(positive_flows.quantile(0.005))
            self.flow_zero_th = max(0.02, min(detected_min_flow, 0.20))
        else:
            self.flow_zero_th = self.cfg.fallback_flow_threshold_lpm

        return (
            df.sort("date")
            .with_columns(
                # Sanificazione glitch di pressione (es. 1e6 -> None -> forward_fill)
                water_pressure=pl.when(
                    pl.col("water_pressure").is_between(
                        self.cfg.min_valid_pressure_mbar, self.cfg.max_valid_pressure_mbar
                    )
                )
                .then(pl.col("water_pressure"))
                .otherwise(None)
                .forward_fill()
                .backward_fill(),
                # Sanificazione glitch di flusso
                flow=pl.when(pl.col("flow").is_between(0.0, self.cfg.max_valid_flow_lpm))
                .then(pl.col("flow"))
                .otherwise(0.0),
                # Flag esplicito se nel secondo specifico c'è stato un glitch hardware (utile per diagnostica sensore)
                is_pressure_glitch=pl.col("water_pressure") > self.cfg.max_valid_pressure_mbar,
            )
            .with_columns(
                # Delta tempo reale in secondi
                dt_sec=pl.col("date")
                .diff()
                .dt.total_seconds()
                .fill_null(1.0)
                .clip(1.0, self.cfg.max_sampling_gap_sec),
                # Sul tubo generale, quando scorre acqua fresca dall'acquedotto, water_temp scende sotto ambient_temp
                cooling_delta=(pl.col("ambient_temp") - pl.col("water_temp")),
                d_pressure=pl.col("water_pressure").diff().fill_null(0.0),
                d_flow=pl.col("flow").diff().fill_null(0.0),
            )
            .with_columns(
                # Volume reale in litri (Flusso l/m diviso 60 moltiplicato per i secondi)
                step_volume_l=pl.col("flow") * (pl.col("dt_sec") / 60.0),
                # Pressione statica di riferimento (quando il flusso è 0, tipicamente ~4345 mbar)
                static_pressure_mbar=pl.when(pl.col("flow") <= self.flow_zero_th)
                .then(pl.col("water_pressure"))
                .otherwise(None)
                .forward_fill()
                .backward_fill(),
            )
            .with_columns(
                # Caduta di pressione dinamica (Delta P = P_statica - P_dinamica)
                pressure_drop_mbar=(pl.col("static_pressure_mbar") - pl.col("water_pressure")).clip(lower_bound=0.0)
            )
        )

    def extract_and_analyze_activities(self, df: pl.DataFrame, n_clusters: int = 6) -> pl.DataFrame:
        """
        Pipeline completa sugli eventi:
        1. Segmentazione con debouncing delle micro-pause.
        2. Estrazione feature fisiche (Idrauliche, Termiche, Cinematiche).
        3. Classificazione deterministica (Regole Fisiche).
        4. Unsupervised Clustering (StandardScaler + KMeans) + Anomaly Detection (IsolationForest).
        """
        df_active = (
            df.with_columns(
                is_flowing=pl.col("flow") > self.flow_zero_th,
                # Pressione massima nei 3 secondi successivi per misurare il colpo d'ariete alla chiusura
                post_3s_max_pressure=pl.col("water_pressure").shift(-3).rolling_max(window_size=5),
            )
            .with_columns(
                last_flow_ts=pl.when(pl.col("is_flowing"))
                .then(pl.col("date"))
                .otherwise(None)
                .forward_fill()
            )
            .with_columns(
                pause_sec=(pl.col("date") - pl.col("last_flow_ts")).dt.total_seconds().fill_null(9999)
            )
            .with_columns(
                # Unisce pause brevi (< 12s) nello stesso evento
                in_event=pl.col("is_flowing") | (pl.col("pause_sec") <= self.cfg.merge_short_pause_sec)
            )
            .with_columns(
                is_start=pl.col("in_event") & (~pl.col("in_event").shift(1).fill_null(False))
            )
            .with_columns(activity_id=pl.col("is_start").cum_sum())
            .filter(pl.col("is_flowing"))
        )

        if df_active.is_empty():
            return pl.DataFrame()

        activities = (
            df_active.group_by("activity_id")
            .agg(
                start_time=pl.col("date").min(),
                end_time=pl.col("date").max(),
                active_duration_sec=pl.col("dt_sec").sum(),
                # Metriche di Volume e Portata
                total_volume_l=pl.col("step_volume_l").sum(),
                peak_flow_lpm=pl.col("flow").max(),
                avg_flow_lpm=pl.col("flow").mean(),
                flow_std=pl.col("flow").std().fill_null(0.0),
                # Cinematica di Apertura/Chiusura e Coda Galleggiante WC
                max_attack_rate=pl.col("d_flow").max(),
                first_half_flow=pl.col("flow").head(pl.len() // 2 + 1).mean(),
                second_half_flow=pl.col("flow").tail(pl.len() // 2 + 1).mean(),
                # Termodinamica sul tubo generale (raffreddamento dovuto all'acqua di rete)
                temp_start=pl.col("water_temp").first(),
                temp_end=pl.col("water_temp").last(),
                temp_min=pl.col("water_temp").min(),
                max_cooling_delta=pl.col("cooling_delta").max(),
                # Idraulica e Pressione (baseline ~4345 mbar)
                static_pressure_before=pl.col("static_pressure_mbar").first(),
                min_dynamic_pressure=pl.col("water_pressure").min(),
                avg_pressure_drop_mbar=pl.col("pressure_drop_mbar").mean(),
                max_pressure_drop_mbar=pl.col("pressure_drop_mbar").max(),
                water_hammer_spike_mbar=(
                        pl.col("post_3s_max_pressure").last() - pl.col("static_pressure_mbar").last()
                ).clip(lower_bound=0.0).fill_null(0.0),
            )
            .with_columns(
                total_span_sec=(pl.col("end_time") - pl.col("start_time")).dt.total_seconds() + 1.0,
                flow_cv=pl.col("flow_std") / pl.col("avg_flow_lpm").clip(lower_bound=0.01),
                tail_ratio=pl.col("second_half_flow") / pl.col("first_half_flow").clip(lower_bound=0.01),
                # Quanti gradi ha perso il tubo generale durante il prelievo
                pipe_temp_drop=(pl.col("temp_start") - pl.col("temp_min")).clip(lower_bound=0.0),
                # Resistenza idraulica apparente: R = dP / Q^2 (mbar / (l/m)^2)
                hydraulic_resistance=pl.col("avg_pressure_drop_mbar") / (
                    pl.col("avg_flow_lpm").pow(2).clip(lower_bound=0.1)),
            )
            .with_columns(
                duty_cycle=(pl.col("active_duration_sec") / pl.col("total_span_sec")).clip(0.0, 1.0)
            )
            .sort("start_time")
        )

        # Rilevamento treni di impulsi da elettrovalvola (Lavatrice / Lavastoviglie)
        activities = (
            activities.with_columns(
                is_solenoid_pulse=(
                        pl.col("total_volume_l").is_between(2.0, 16.0)
                        & (pl.col("flow_cv") < 0.35)
                        & (pl.col("max_attack_rate") > 2.0)
                )
            )
            .with_columns(
                solenoid_train_count=pl.col("is_solenoid_pulse")
                .cast(pl.Int32)
                .rolling_sum_by("start_time", window_size=f"{self.cfg.appliance_window_sec}s")
            )
        )

        # 1. Classificazione Fisica Deterministica
        activities = self._apply_physical_rules(activities)

        # 2. Machine Learning: Clustering (KMeans) + Anomaly Detection (IsolationForest)
        activities = self._apply_ml_models(activities, n_clusters=n_clusters)

        return activities

    def _apply_physical_rules(self, activities: pl.DataFrame) -> pl.DataFrame:
        """Regole basate su cinematica del flusso, caduta di pressione e raffreddamento del tubo generale."""
        return activities.with_columns(
            physical_category=pl.when(
                (pl.col("active_duration_sec") > 900) & (pl.col("total_volume_l") > 80) & (pl.col("flow_cv") < 0.22)
            )
            .then(pl.lit("Irrigation / Garden Hose"))
            .when(
                (pl.col("total_volume_l") > 75)
                & pl.col("active_duration_sec").is_between(240, 1200)
                & (pl.col("duty_cycle") > 0.90)
            )
            .then(pl.lit("Bath Filling"))
            .when(pl.col("is_solenoid_pulse") & (pl.col("solenoid_train_count") >= 3))
            .then(pl.lit("Appliance (Washing Machine / Dishwasher)"))
            .when(
                pl.col("total_volume_l").is_between(22.0, 95.0)
                & pl.col("active_duration_sec").is_between(150, 1200)
                & pl.col("avg_flow_lpm").is_between(5.0, 15.0)
                & (pl.col("pipe_temp_drop") > 0.4)  # Il flusso prolungato raffredda sensibilmente il tubo generale
            )
            .then(pl.lit("Shower"))
            .when(
                pl.col("total_volume_l").is_between(3.5, 10.5)
                & pl.col("active_duration_sec").is_between(20, 110)
                & (pl.col("duty_cycle") > 0.92)
                & ((pl.col("tail_ratio") < 0.85) | (pl.col("flow_cv") < 0.28))
            )
            .then(pl.lit("Toilet Flush"))
            .when((pl.col("total_volume_l") < 2.0) & (pl.col("active_duration_sec") <= 30))
            .then(pl.lit("Quick Tap (Hands / Glass)"))
            .when(pl.col("total_volume_l").is_between(2.0, 22.0) & (pl.col("active_duration_sec") <= 360))
            .then(pl.lit("Sink / Kitchen Wash"))
            .otherwise(pl.lit("Other / Unclassified"))
        )

    def _apply_ml_models(self, activities: pl.DataFrame, n_clusters: int = 6) -> pl.DataFrame:
        """
        Applica:
        - StandardScaler + KMeans per trovare cluster comportamentali.
        - RobustScaler + IsolationForest per calcolare uno score di anomalia su ogni evento.
        """
        if len(activities) < max(10, n_clusters):
            return activities.with_columns(
                usage_cluster=pl.lit(0),
                is_ml_anomaly=pl.lit(False),
                anomaly_score=pl.lit(0.0),
            )

        ml_features = [
            "active_duration_sec",
            "total_volume_l",
            "peak_flow_lpm",
            "flow_cv",
            "avg_pressure_drop_mbar",
            "hydraulic_resistance",
            "pipe_temp_drop",
            "tail_ratio",
        ]

        # Estrazione matrice Numpy pulita da eventuali NaN/Inf
        X = activities.select(ml_features).fill_nan(0.0).fill_null(0.0).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. Clustering con StandardScaler + KMeans
        # Applichiamo log1p a durata e volume per ridurre la skewness prima di KMeans
        X_cluster = X.copy()
        X_cluster[:, 0] = np.log1p(X_cluster[:, 0])  # log(duration)
        X_cluster[:, 1] = np.log1p(X_cluster[:, 1])  # log(volume)

        scaler_km = StandardScaler()
        X_scaled_km = scaler_km.fit_transform(X_cluster)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled_km)

        # 2. Anomaly Detection con RobustScaler + IsolationForest
        # IsolationForest individua eventi con combinazioni fisiche anomale (es. enorme caduta di pressione
        # a basso flusso, o durate/volumi fuori distribuzione per l'abitazione)
        scaler_iso = RobustScaler()
        X_scaled_iso = scaler_iso.fit_transform(X)

        iso_forest = IsolationForest(
            contamination=self.cfg.isolation_forest_contamination,
            random_state=42,
            n_estimators=150,
        )
        # fit_predict restituisce -1 per le anomalie e 1 per i punti normali
        iso_preds = iso_forest.fit_predict(X_scaled_iso)
        # decision_function: più è negativo, più l'evento è anomalo
        iso_scores = iso_forest.decision_function(X_scaled_iso)

        return activities.with_columns(
            pl.Series("usage_cluster", cluster_labels),
            pl.Series("is_ml_anomaly", iso_preds == -1),
            pl.Series("anomaly_score", -iso_scores),  # Invertito: più è alto, più è anomalo
        )

    def detect_leaks_and_system_anomalies(
            self, df: pl.DataFrame, activities: pl.DataFrame
    ) -> dict[str, pl.DataFrame]:
        """
        Suite completa di Leak Detection (Fisica + Termica + IsolationForest Giornaliero):
        1. MNF + Thermal Signature: Perdita continua notturna confermata dal mancato riscaldamento del tubo.
        2. Flapper Leak: Rabbocchi periodici cassetta WC.
        3. Sub-Threshold Pressure Decay: Caduta di pressione a flusso zero.
        4. Daily IsolationForest: Giorni con pattern complessivo anomalo.
        """
        # 1. Analisi Giornaliera e Notturna (02:00 - 05:00) + Firma Termica sul Tubo Generale
        night_df = (
            df.with_columns(
                day=pl.col("date").dt.date(),
                hour=pl.col("date").dt.hour(),
            )
            .filter(pl.col("hour").is_between(self.cfg.mnf_window_hours[0], self.cfg.mnf_window_hours[1] - 1))
            .group_by("day")
            .agg(
                min_night_flow_lpm=pl.col("flow").quantile(0.02),
                median_night_flow_lpm=pl.col("flow").median(),
                zero_flow_ratio=(pl.col("flow") <= self.flow_zero_th).mean(),
                night_static_pressure_mbar=pl.col("water_pressure").median(),
                # TRUCCO TERMICO SUL TUBO GENERALE:
                # Se di notte non ci sono perdite, l'acqua ferma nel tubo raggiunge ambient_temp (cooling_delta ~ 0).
                # Se c'è anche un micro-gocciolamento invisibile al flussimetro, entra acqua fredda
                # dall'acquedotto e il tubo rimane più freddo dell'ambiente (cooling_delta > 1.5°C)!
                night_unexplained_cooling_c=pl.col("cooling_delta").median(),
                pressure_glitch_count=pl.col("is_pressure_glitch").sum(),
            )
            .sort("day")
        )

        # Aggiungiamo IsolationForest sui profili notturni se abbiamo almeno 7 giorni di storico
        if len(night_df) >= 7:
            night_features = [
                "min_night_flow_lpm",
                "zero_flow_ratio",
                "night_static_pressure_mbar",
                "night_unexplained_cooling_c",
            ]
            X_night = night_df.select(night_features).fill_nan(0.0).fill_null(0.0).to_numpy()
            iso_night = IsolationForest(contamination=0.10, random_state=42)
            night_preds = iso_night.fit_predict(RobustScaler().fit_transform(X_night))
            night_df = night_df.with_columns(pl.Series("is_anomalous_night_ml", night_preds == -1))
        else:
            night_df = night_df.with_columns(is_anomalous_night_ml=pl.lit(False))

        continuous_leaks = night_df.filter(
            (pl.col("min_night_flow_lpm") > self.flow_zero_th)
            | (pl.col("zero_flow_ratio") < 0.20)
            | ((pl.col("zero_flow_ratio") > 0.90) & (pl.col("night_unexplained_cooling_c") > 2.0))
            | pl.col("is_anomalous_night_ml")
        ).with_columns(
            estimated_leak_l_per_day=(pl.col("min_night_flow_lpm") * 60.0 * 24.0),
            leak_diagnosis=pl.when(pl.col("min_night_flow_lpm") > self.flow_zero_th)
            .then(pl.lit("Continuous Background Leak (Flow never hits zero)"))
            .when(pl.col("night_unexplained_cooling_c") > 2.0)
            .then(pl.lit("Suspected Sub-Threshold Micro-Leak (Pipe stays cold at night)"))
            .otherwise(pl.lit("Anomalous Night Pattern (Detected by IsolationForest)")),
        )

        # 2. Perdita Ciclica Cassetta WC (Flapper Leak)
        flapper_leaks = pl.DataFrame()
        if not activities.is_empty():
            flapper_leaks = (
                activities.filter(
                    pl.col("total_volume_l").is_between(0.2, 3.0)
                    & pl.col("active_duration_sec").is_between(3, 40)
                )
                .with_columns(
                    inter_arrival_min=pl.col("start_time").diff().dt.total_seconds() / 60.0
                )
                .with_columns(
                    roll_mean_min=pl.col("inter_arrival_min").rolling_mean(window_size=4),
                    roll_std_min=pl.col("inter_arrival_min").rolling_std(window_size=4),
                )
                .with_columns(interval_cv=pl.col("roll_std_min") / pl.col("roll_mean_min"))
                .filter(
                    (pl.col("interval_cv") < 0.25) & pl.col("roll_mean_min").is_between(4.0, 240.0)
                )
            )

        # 3. Eventi Critici (Burst, Rubinetti dimenticati, Colpi d'Ariete e Anomalie IsolationForest)
        critical_events = pl.DataFrame()
        if not activities.is_empty():
            critical_events = activities.filter(
                (pl.col("active_duration_sec") > self.cfg.burst_duration_sec)
                | (pl.col("water_hammer_spike_mbar") > self.cfg.water_hammer_spike_mbar)
                | pl.col("is_ml_anomaly")
            ).with_columns(
                alert_reason=pl.when(pl.col("active_duration_sec") > self.cfg.burst_duration_sec)
                .then(pl.lit("CRITICAL: Continuous Flow > 40m (Left Open / Pipe Burst)"))
                .when(pl.col("water_hammer_spike_mbar") > self.cfg.water_hammer_spike_mbar)
                .then(pl.lit("WARNING: Severe Water Hammer Spike (> +800 mbar)"))
                .otherwise(pl.lit("ANOMALY: Unusual Hydraulic/Kinematic Signature (IsolationForest)"))
            )

        return {
            "night_health_and_continuous_leaks": continuous_leaks,
            "cyclic_wc_flapper_leaks": flapper_leaks,
            "critical_and_ml_anomalous_events": critical_events,
        }

    def summarize_disaggregation(self, activities: pl.DataFrame) -> pl.DataFrame:
        """Restituisce il breakdown dei consumi per categoria fisica (Volume totale, %, Durata media, Pressione)."""
        if activities.is_empty():
            return pl.DataFrame()

        total_vol = activities["total_volume_l"].sum()
        return (
            activities.group_by("physical_category")
            .agg(
                event_count=pl.len(),
                total_liters=pl.col("total_volume_l").sum().round(1),
                avg_liters_per_use=pl.col("total_volume_l").mean().round(2),
                avg_duration_sec=pl.col("active_duration_sec").mean().round(1),
                avg_flow_lpm=pl.col("avg_flow_lpm").mean().round(2),
                avg_pressure_drop_mbar=pl.col("avg_pressure_drop_mbar").mean().round(0),
                ml_anomalies_count=pl.col("is_ml_anomaly").sum(),
            )
            .with_columns(
                volume_share_pct=((pl.col("total_liters") / max(total_vol, 0.01)) * 100.0).round(1)
            )
            .sort("total_liters", descending=True)
        )


if __name__ == "__main__":
    service = WaterAnalyticsService()
    df = service.load_water(
        device_id="EWCS30065",
        start=datetime.datetime(2026, 4, 1),
        end=datetime.datetime(2026, 4, 30),
    )
    activities = service.extract_and_analyze_activities(df, n_clusters=6)
    summary = service.summarize_disaggregation(activities)
    leaks = service.detect_leaks_and_system_anomalies(df, activities)

    print("\n--- RIEPILOGO DISAGGREGAZIONE CONSUMI ---")
    print(summary)
    print("\n--- DIAGNOSI PERDITE NOTTURNE & MNF ---")
    print(leaks["night_health_and_continuous_leaks"])
    print("\n--- EVENTI ANOMALI (ML / BURST / WATER HAMMER) ---")
    print(leaks["critical_and_ml_anomalous_events"].select([
        "start_time", "physical_category", "total_volume_l", "active_duration_sec",
        "avg_pressure_drop_mbar", "water_hammer_spike_mbar", "anomaly_score", "alert_reason"
    ]))