"""
Trip Cost Predictor — Railway Data Processor
============================================
This module processes raw railway datasets (price_data.csv and schedules.csv)
and generates clean, production-ready railway datasets:
1. data/railway_stations.csv
2. data/railway_fares_clean.csv
3. data/destination_railway_mapping.csv

Safety Constraints:
- Does NOT modify original raw data files.
- Preserves fare record granularity without synthetic generation.
"""

import os
import sys
import ast
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("railway_processor")

# Path configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

PRICE_DATA_PATH = DATA_DIR / "price_data.csv"
SCHEDULES_PATH = DATA_DIR / "schedules.csv"

OUTPUT_STATIONS_PATH = DATA_DIR / "railway_stations.csv"
OUTPUT_FARES_PATH = DATA_DIR / "railway_fares_clean.csv"
OUTPUT_MAPPING_PATH = DATA_DIR / "destination_railway_mapping.csv"

# Predefined destination mappings for the 35 project destinations
DESTINATION_MAPPINGS = [
    # Goa
    ("Goa", "MAO", "DIRECT"),
    ("Goa", "KRMI", "DIRECT"),
    ("Goa", "THVM", "DIRECT"),
    # Mumbai
    ("Mumbai", "MMCT", "DIRECT"),
    ("Mumbai", "CSMT", "DIRECT"),
    ("Mumbai", "DR", "DIRECT"),
    ("Mumbai", "BDTS", "DIRECT"),
    # Pune
    ("Pune", "PUNE", "DIRECT"),
    # Lonavala
    ("Lonavala", "LNL", "DIRECT"),
    # Mahabaleshwar (Gateways: Satara, Pune)
    ("Mahabaleshwar", "STR", "GATEWAY"),
    ("Mahabaleshwar", "PUNE", "GATEWAY"),
    # Nashik
    ("Nashik", "NK", "DIRECT"),
    # Ahmedabad
    ("Ahmedabad", "ADI", "DIRECT"),
    # Udaipur
    ("Udaipur", "UDZ", "DIRECT"),
    # Delhi
    ("Delhi", "NDLS", "DIRECT"),
    ("Delhi", "DLI", "DIRECT"),
    ("Delhi", "NZM", "DIRECT"),
    ("Delhi", "ANVT", "DIRECT"),
    # Agra
    ("Agra", "AGC", "DIRECT"),
    ("Agra", "AF", "DIRECT"),
    # Jaipur
    ("Jaipur", "JP", "DIRECT"),
    # Jodhpur
    ("Jodhpur", "JU", "DIRECT"),
    # Amritsar
    ("Amritsar", "ASR", "DIRECT"),
    # Manali (Gateways: Chandigarh, Kalka)
    ("Manali", "CDG", "GATEWAY"),
    ("Manali", "KLK", "GATEWAY"),
    # Shimla (Gateways: Kalka, Chandigarh)
    ("Shimla", "KLK", "GATEWAY"),
    ("Shimla", "CDG", "GATEWAY"),
    # Rishikesh
    ("Rishikesh", "YNRK", "DIRECT"),
    ("Rishikesh", "RKSH", "DIRECT"),
    # Haridwar
    ("Haridwar", "HW", "DIRECT"),
    # Chandigarh
    ("Chandigarh", "CDG", "DIRECT"),
    # Bengaluru
    ("Bengaluru", "SBC", "DIRECT"),
    ("Bengaluru", "SMVB", "DIRECT"),
    ("Bengaluru", "YPR", "DIRECT"),
    # Hyderabad
    ("Hyderabad", "HYB", "DIRECT"),
    ("Hyderabad", "SC", "DIRECT"),
    # Chennai
    ("Chennai", "MAS", "DIRECT"),
    ("Chennai", "MS", "DIRECT"),
    # Kochi
    ("Kochi", "ERS", "DIRECT"),
    ("Kochi", "ERN", "DIRECT"),
    # Munnar (Gateways: Ernakulam, Aluva)
    ("Munnar", "ERS", "GATEWAY"),
    ("Munnar", "AWY", "GATEWAY"),
    # Alleppey
    ("Alleppey", "ALLP", "DIRECT"),
    # Ooty (Gateways: Coimbatore, Mettupalayam)
    ("Ooty", "CBE", "GATEWAY"),
    ("Ooty", "MTP", "GATEWAY"),
    # Coorg (Gateway: Mysuru)
    ("Coorg", "MYS", "GATEWAY"),
    # Wayanad (Gateway: Kozhikode)
    ("Wayanad", "CLT", "GATEWAY"),
    # Pondicherry
    ("Pondicherry", "PDY", "DIRECT"),
    # Kolkata
    ("Kolkata", "HWH", "DIRECT"),
    ("Kolkata", "SDAH", "DIRECT"),
    ("Kolkata", "KOAA", "DIRECT"),
    # Darjeeling (Gateways: New Jalpaiguri, Siliguri)
    ("Darjeeling", "NJP", "GATEWAY"),
    ("Darjeeling", "SGUJ", "GATEWAY"),
    # Gangtok (Gateways: New Jalpaiguri, Siliguri)
    ("Gangtok", "NJP", "GATEWAY"),
    ("Gangtok", "SGUJ", "GATEWAY"),
    # Shillong (Gateway: Guwahati)
    ("Shillong", "GHY", "GATEWAY"),
    # Guwahati
    ("Guwahati", "GHY", "DIRECT"),
    # Varanasi
    ("Varanasi", "BSB", "DIRECT"),
    # Mysuru
    ("Mysuru", "MYS", "DIRECT")
]


def load_raw_datasets(price_path: Path, schedules_path: Path):
    """Loads raw railway CSV datasets in read-only mode."""
    logger.info("Reading raw datasets...")
    if not price_path.exists():
        raise FileNotFoundError(f"Missing price dataset: {price_path}")
    if not schedules_path.exists():
        raise FileNotFoundError(f"Missing schedules dataset: {schedules_path}")

    price_df = pd.read_csv(price_path)
    schedules_df = pd.read_csv(schedules_path)
    logger.info("Loaded price_data (%d rows) and schedules (%d rows).", len(price_df), len(schedules_df))
    return price_df, schedules_df


def extract_station_master(schedules_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Parses schedules.stationList to extract stationCode and stationName.
    Normalizes and removes duplicates, writing data/railway_stations.csv.
    """
    logger.info("Parsing stationList from schedules to build station master...")
    station_dict = {}

    for row_idx, s_list_str in enumerate(schedules_df["stationList"]):
        if not isinstance(s_list_str, str) or not s_list_str.strip():
            continue
        try:
            stations = ast.literal_eval(s_list_str)
            if not isinstance(stations, list):
                continue
            for stn in stations:
                if isinstance(stn, dict):
                    code = str(stn.get("stationCode", "")).strip().upper()
                    name = str(stn.get("stationName", "")).strip().upper()
                    if code and code not in station_dict:
                        station_dict[code] = name
        except Exception as ex:
            logger.warning("Error parsing stationList at row %d: %s", row_idx, ex)

    stations_df = pd.DataFrame(
        [{"station_code": code, "station_name": name} for code, name in station_dict.items()]
    )
    stations_df.sort_values(by="station_code", inplace=True)
    stations_df.drop_duplicates(subset=["station_code"], inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stations_df.to_csv(output_path, index=False)
    logger.info("Saved railway_stations.csv with %d unique stations.", len(stations_df))
    return stations_df


def clean_fare_data(price_df: pd.DataFrame, schedules_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Cleans price_data.csv:
    1. Corrects negative duration records (midnight rollover % 1440).
    2. Deduplicates multiple snapshot queries using latest timeStamp.
    3. Joins with schedules.csv on trainNumber to add train_name.
    4. Selects and validates clean schema.
    5. Saves to data/railway_fares_clean.csv.
    """
    logger.info("Cleaning fare data...")
    df = price_df.copy()
    initial_rows = len(df)

    # 1. Correct negative durations caused by midnight rollover
    neg_mask = df["duration"] < 0
    corrected_count = int(neg_mask.sum())
    logger.info("Correcting %d negative duration records via modulo 1440...", corrected_count)
    df["duration_mins"] = df["duration"].apply(
        lambda d: int(round(d % 1440)) if d < 0 else int(round(d))
    )

    # 2. Handle duplicate fare snapshots: prefer latest timeStamp
    logical_key = ["trainNumber", "fromStnCode", "toStnCode", "classCode"]
    logger.info("Sorting and deduplicating logical fare keys %s...", logical_key)
    df["parsed_time"] = pd.to_datetime(df["timeStamp"], errors="coerce")
    
    # Sort with parsed_time ascending so keep='last' picks latest snapshot
    # Records with null/invalid timestamp are naturally placed at the start as fallback
    df_sorted = df.sort_values(by=logical_key + ["parsed_time"])
    df_dedup = df_sorted.drop_duplicates(subset=logical_key, keep="last").copy()
    dedup_rows = len(df_dedup)
    logger.info("Deduplication complete: %d -> %d records (%d snapshots removed).",
                initial_rows, dedup_rows, initial_rows - dedup_rows)

    # 3. Join with schedules.csv on trainNumber to append train_name
    logger.info("Joining fare records with schedules on trainNumber...")
    s_trains = schedules_df[["trainNumber", "trainName"]].drop_duplicates(subset=["trainNumber"]).copy()
    s_trains.rename(columns={"trainName": "train_name"}, inplace=True)

    merged = pd.merge(df_dedup, s_trains, on="trainNumber", how="left")

    # 4. Column mapping and final selection
    rename_cols = {
        "trainNumber": "train_number",
        "fromStnCode": "from_station_code",
        "toStnCode": "to_station_code",
        "classCode": "travel_class",
        "distance": "distance_km",
        "baseFare": "base_fare",
        "totalFare": "total_fare",
        "superfastCharge": "superfast_charge",
        "serviceTax": "service_tax",
        "cateringCharge": "catering_charge",
        "dynamicFare": "dynamic_fare"
    }

    final_columns = [
        "train_number",
        "train_name",
        "from_station_code",
        "to_station_code",
        "travel_class",
        "distance_km",
        "duration_mins",
        "base_fare",
        "total_fare",
        "superfast_charge",
        "service_tax",
        "catering_charge",
        "dynamic_fare"
    ]

    clean_fares_df = merged.rename(columns=rename_cols)[final_columns].copy()

    # 5. Data integrity assertions
    assert (clean_fares_df["distance_km"] < 0).sum() == 0, "Found negative distances!"
    assert (clean_fares_df["duration_mins"] < 0).sum() == 0, "Found negative durations!"
    assert (clean_fares_df["total_fare"] < 0).sum() == 0, "Found negative total fares!"
    assert clean_fares_df["train_number"].isnull().sum() == 0, "Missing train numbers!"
    assert clean_fares_df["from_station_code"].isnull().sum() == 0, "Missing from_station_code!"
    assert clean_fares_df["to_station_code"].isnull().sum() == 0, "Missing to_station_code!"
    assert clean_fares_df["train_name"].isnull().sum() == 0, "Missing train names!"
    assert clean_fares_df.duplicated(subset=["train_number", "from_station_code", "to_station_code", "travel_class"]).sum() == 0, "Duplicate logical fare records found!"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_fares_df.to_csv(output_path, index=False)
    logger.info("Saved railway_fares_clean.csv with %d records.", len(clean_fares_df))
    return clean_fares_df


def generate_destination_mapping(stations_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Creates destination_railway_mapping.csv for the 35 destinations.
    Validates that every mapped station exists in railway_stations.csv.
    """
    logger.info("Building destination railway mapping...")
    valid_station_dict = dict(zip(stations_df["station_code"], stations_df["station_name"]))

    mapping_rows = []
    for dest, stn_code, map_type in DESTINATION_MAPPINGS:
        stn_code_upper = stn_code.strip().upper()
        if stn_code_upper not in valid_station_dict:
            raise ValueError(f"Station code '{stn_code_upper}' for destination '{dest}' not found in station master!")

        mapping_rows.append({
            "destination": dest,
            "station_code": stn_code_upper,
            "station_name": valid_station_dict[stn_code_upper],
            "mapping_type": map_type
        })

    mapping_df = pd.DataFrame(mapping_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(output_path, index=False)
    logger.info("Saved destination_railway_mapping.csv with %d mappings.", len(mapping_df))
    return mapping_df


def run_pipeline():
    """Executes the full railway data layer pipeline and validates outputs."""
    logger.info("=== STARTING RAILWAY DATA LAYER PROCESSING ===")
    price_df, schedules_df = load_raw_datasets(PRICE_DATA_PATH, SCHEDULES_PATH)

    # Step 1 & 2: Station Master
    stations_df = extract_station_master(schedules_df, OUTPUT_STATIONS_PATH)

    # Step 3, 4, 5, 6, 7: Clean Fares Dataset
    clean_fares_df = clean_fare_data(price_df, schedules_df, OUTPUT_FARES_PATH)

    # Step 8: Destination Mapping
    mapping_df = generate_destination_mapping(stations_df, OUTPUT_MAPPING_PATH)

    logger.info("=== RAILWAY DATA LAYER PROCESSING FINISHED SUCCESSFULLY ===")
    return {
        "price_df": price_df,
        "schedules_df": schedules_df,
        "stations_df": stations_df,
        "clean_fares_df": clean_fares_df,
        "mapping_df": mapping_df
    }


if __name__ == "__main__":
    run_pipeline()
