"""
Trip Cost Predictor — Railway Fare Service
==========================================
Provides static-dataset-based railway fare calculation and lookup for the
Trip Cost Predictor project.

Features:
- Case-insensitive destination matching using destination_railway_mapping.csv.
- Exact OD-pair fare lookup from railway_fares_clean.csv.
- Robust median-fallback estimation when origin station is omitted.
- Full validation for travel classes, traveler counts, stations, and trains.
- Zero external API dependencies (works strictly offline from clean data).
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class RailwayFareService:
    """
    Service for calculating and querying railway fares across project destinations.
    Loads and caches clean datasets once upon instantiation.
    """

    SUPPORTED_CLASSES = {"1A", "2A", "3A", "SL", "2S", "CC"}

    def __init__(
        self,
        fares_path: Optional[Path] = None,
        mapping_path: Optional[Path] = None,
        stations_path: Optional[Path] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "data"

        self.fares_path = Path(fares_path or data_dir / "railway_fares_clean.csv")
        self.mapping_path = Path(mapping_path or data_dir / "destination_railway_mapping.csv")
        self.stations_path = Path(stations_path or data_dir / "railway_stations.csv")

        self._load_datasets()

    def _load_datasets(self) -> None:
        """Loads and indexes datasets once upon initialization."""
        if not self.fares_path.exists():
            raise FileNotFoundError(f"Missing clean fares dataset: {self.fares_path}")
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Missing destination mapping dataset: {self.mapping_path}")
        if not self.stations_path.exists():
            raise FileNotFoundError(f"Missing station master dataset: {self.stations_path}")

        self.fares_df = pd.read_csv(self.fares_path)
        self.mapping_df = pd.read_csv(self.mapping_path)
        self.stations_df = pd.read_csv(self.stations_path)

        # Normalize casing in mapping
        self.mapping_df["dest_lower"] = self.mapping_df["destination"].astype(str).str.strip().str.lower()
        self.mapping_df["station_code"] = self.mapping_df["station_code"].astype(str).str.strip().str.upper()

        # Station name dictionary for fast lookup
        self.station_names = dict(
            zip(
                self.stations_df["station_code"].astype(str).str.strip().str.upper(),
                self.stations_df["station_name"].astype(str).str.strip(),
            )
        )

        # Pre-group mapping by lowercase destination
        self._dest_to_stations: Dict[str, List[Dict[str, str]]] = {}
        for _, row in self.mapping_df.iterrows():
            d_key = row["dest_lower"]
            entry = {
                "destination": row["destination"],
                "station_code": row["station_code"],
                "station_name": row["station_name"],
                "mapping_type": row["mapping_type"],
            }
            if d_key not in self._dest_to_stations:
                self._dest_to_stations[d_key] = []
            self._dest_to_stations[d_key].append(entry)

    def normalize_travel_class(self, travel_class: str) -> str:
        """
        Normalizes and validates railway travel class code.
        Raises ValueError if travel class is invalid.
        """
        if not travel_class or not isinstance(travel_class, str):
            raise ValueError(
                f"Invalid travel class: {travel_class}. Supported classes: {', '.join(sorted(self.SUPPORTED_CLASSES))}."
            )
        normalized = travel_class.strip().upper()
        if normalized not in self.SUPPORTED_CLASSES:
            raise ValueError(
                f"Unsupported travel class '{travel_class}'. Supported classes are: {', '.join(sorted(self.SUPPORTED_CLASSES))}."
            )
        return normalized

    def get_destination_stations(self, destination: str) -> List[Dict[str, str]]:
        """
        Retrieves mapped stations for a destination (case-insensitive).
        Returns a list of station dictionaries or raises ValueError if destination is unmapped.
        """
        if not destination or not isinstance(destination, str):
            raise ValueError(f"Invalid destination: {destination}.")

        d_key = destination.strip().lower()
        if d_key not in self._dest_to_stations:
            available_dests = sorted(list(self.mapping_df["destination"].unique()))
            raise ValueError(
                f"Destination '{destination}' not found in destination mapping. "
                f"Available destinations: {len(available_dests)} total."
            )

        return self._dest_to_stations[d_key]

    def find_fares(
        self,
        destination: str,
        travel_class: str,
        origin_station: Optional[str] = None,
        train_number: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Finds matching fare records from railway_fares_clean.csv.
        Filters by destination stations, travel class, and optional origin and train number.
        """
        norm_class = self.normalize_travel_class(travel_class)
        dest_entries = self.get_destination_stations(destination)
        dest_codes = [e["station_code"] for e in dest_entries]

        mask = (self.fares_df["to_station_code"].isin(dest_codes)) & (
            self.fares_df["travel_class"] == norm_class
        )

        if origin_station is not None and str(origin_station).strip():
            norm_origin = str(origin_station).strip().upper()
            mask = mask & (self.fares_df["from_station_code"] == norm_origin)

        if train_number is not None:
            try:
                t_num = int(train_number)
                mask = mask & (self.fares_df["train_number"] == t_num)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid train number '{train_number}'. Must be an integer.")

        return self.fares_df[mask].copy()

    def calculate_railway_cost(
        self,
        destination: str,
        travelers: int,
        travel_class: str = "SL",
        origin_station: Optional[str] = None,
        train_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Calculates total railway travel cost for a trip.
        Supports both exact OD lookups and dataset median fallbacks.
        Returns a structured dictionary with full cost and route metadata.
        """
        # 1. Validate travelers count
        try:
            num_travelers = int(travelers)
            if num_travelers <= 0:
                return {
                    "status": "ERROR",
                    "error": f"Invalid travelers value: {travelers}. Must be a positive integer >= 1.",
                }
        except (ValueError, TypeError):
            return {
                "status": "ERROR",
                "error": f"Invalid travelers value: '{travelers}'. Must be a positive integer.",
            }

        # 2. Validate travel class
        try:
            norm_class = self.normalize_travel_class(travel_class)
        except ValueError as ex:
            return {"status": "ERROR", "error": str(ex)}

        # 3. Validate destination
        try:
            dest_stations = self.get_destination_stations(destination)
        except ValueError as ex:
            return {"status": "ERROR", "error": str(ex)}

        canonical_dest = dest_stations[0]["destination"]
        # Partition stations by mapping type (DIRECT vs GATEWAY)
        direct_stns = [s for s in dest_stations if s.get("mapping_type") == "DIRECT"]
        gateway_stns = [s for s in dest_stations if s.get("mapping_type") == "GATEWAY"]
        direct_codes = [s["station_code"] for s in direct_stns]
        gateway_codes = [s["station_code"] for s in gateway_stns]

        # 4. Validate origin station (if provided)
        norm_origin = None
        if origin_station is not None and str(origin_station).strip():
            norm_origin = str(origin_station).strip().upper()

        # 5. Validate train number (if provided)
        parsed_train_num = None
        if train_number is not None and str(train_number).strip():
            try:
                parsed_train_num = int(train_number)
            except (ValueError, TypeError):
                return {
                    "status": "ERROR",
                    "error": f"Invalid train number '{train_number}'. Must be an integer.",
                }

        # -------------------------------------------------------------
        # CASE A: Origin Station Provided -> Attempt EXACT then GATEWAY
        # -------------------------------------------------------------
        if norm_origin is not None:
            selected = None
            is_gateway = False
            is_reverse = False

            # Priority 1: EXACT FORWARD DIRECT ROUTE
            if direct_codes:
                direct_matches = self._query_fare_records(
                    dest_codes=direct_codes,
                    norm_class=norm_class,
                    norm_origin=norm_origin,
                    train_num=parsed_train_num,
                )
                if not direct_matches.empty:
                    selected = direct_matches.iloc[0]
                    is_gateway = False
                    is_reverse = False

            # Priority 2: VALID GATEWAY FORWARD ROUTE
            if selected is None and gateway_codes:
                gateway_matches = self._query_fare_records(
                    dest_codes=gateway_codes,
                    norm_class=norm_class,
                    norm_origin=norm_origin,
                    train_num=parsed_train_num,
                )
                if not gateway_matches.empty:
                    selected = gateway_matches.iloc[0]
                    is_gateway = True
                    is_reverse = False

            # Priority 3: REVERSE-ROUTE DATASET ESTIMATE (DIRECT STATIONS)
            if selected is None and direct_codes:
                rev_direct_matches = self._query_reverse_fare_records(
                    dest_codes=direct_codes,
                    norm_class=norm_class,
                    norm_origin=norm_origin,
                    train_num=parsed_train_num,
                )
                if not rev_direct_matches.empty:
                    selected = rev_direct_matches.iloc[0]
                    is_gateway = False
                    is_reverse = True

            # Priority 3b: REVERSE-ROUTE DATASET ESTIMATE (GATEWAY STATIONS)
            if selected is None and gateway_codes:
                rev_gateway_matches = self._query_reverse_fare_records(
                    dest_codes=gateway_codes,
                    norm_class=norm_class,
                    norm_origin=norm_origin,
                    train_num=parsed_train_num,
                )
                if not rev_gateway_matches.empty:
                    selected = rev_gateway_matches.iloc[0]
                    is_gateway = True
                    is_reverse = True

            # If a valid record was selected (forward, gateway, or reverse)
            if selected is not None:
                fare_per_person = int(selected["total_fare"])
                total_cost = int(fare_per_person * num_travelers)

                if is_reverse:
                    dest_code = str(selected["from_station_code"])
                    dest_name = self.station_names.get(dest_code, dest_code)
                    from_code = str(selected["to_station_code"])
                    from_name = self.station_names.get(from_code, from_code)
                    fare_source = "REVERSE_ROUTE_DATASET_ESTIMATE"
                    route_in_dataset = f"{dest_name} ({dest_code}) → {from_name} ({from_code})"
                    reverse_warning = (
                        f"Direct {from_name} → {canonical_dest} fare was not available in the project dataset. "
                        f"This estimate uses an observed reverse-direction railway fare."
                    )
                else:
                    dest_code = str(selected["to_station_code"])
                    dest_name = self.station_names.get(dest_code, dest_code)
                    from_code = str(selected["from_station_code"])
                    from_name = self.station_names.get(from_code, from_code)
                    fare_source = "GATEWAY_DATASET_ESTIMATE" if is_gateway else "EXACT_DATASET_FARE"
                    route_in_dataset = None
                    reverse_warning = None

                gateway_note = (
                    f"Direct railway fare to {canonical_dest} was not found in the project dataset. "
                    f"A gateway station estimate ({dest_name}) is shown."
                    if (is_gateway and not is_reverse)
                    else None
                )
                gateway_disclaimer = (
                    "Gateway estimate does not include the onward road/local transfer from the gateway station to the destination."
                    if is_gateway
                    else None
                )

                return {
                    "status": "SUCCESS",
                    "destination": canonical_dest,
                    "travel_class": norm_class,
                    "origin_station": from_code,
                    "origin_station_name": from_name,
                    "destination_station": dest_code,
                    "destination_station_name": dest_name,
                    "is_gateway": is_gateway,
                    "gateway_station": dest_code if is_gateway else None,
                    "gateway_station_name": dest_name if is_gateway else None,
                    "gateway_note": gateway_note,
                    "gateway_disclaimer": gateway_disclaimer,
                    "is_reverse": is_reverse,
                    "route_in_dataset": route_in_dataset,
                    "reverse_warning": reverse_warning,
                    "train_number": int(selected["train_number"]),
                    "train_name": str(selected["train_name"]),
                    "fare_per_person": fare_per_person,
                    "travelers": num_travelers,
                    "total_railway_cost": total_cost,
                    "distance_km": int(selected["distance_km"]),
                    "duration_mins": int(selected["duration_mins"]),
                    "fare_source": fare_source,
                }

            # Priority 5: NO FARE AVAILABLE FOR SPECIFIED ORIGIN
            return {
                "status": "ERROR",
                "error": (
                    f"No railway fare was found in the project dataset for this origin and destination. "
                    f"Direct railway fare was not found for origin '{norm_origin}' and destination '{canonical_dest}' in the project dataset. "
                    f"Try another origin station or leave origin unspecified to use the dataset median estimate."
                ),
            }

        # -------------------------------------------------------------
        # CASE B: Origin Station NOT Provided -> DATASET MEDIAN ESTIMATE
        # -------------------------------------------------------------
        matches = self.find_fares(
            destination=canonical_dest,
            travel_class=norm_class,
            origin_station=None,
            train_number=parsed_train_num,
        )

        if matches.empty:
            available_classes_for_dest = self.fares_df[
                self.fares_df["to_station_code"].isin(dest_codes)
            ]["travel_class"].unique()
            class_info = (
                f"Available classes for {canonical_dest}: {', '.join(sorted(available_classes_for_dest))}"
                if len(available_classes_for_dest) > 0
                else f"No trains currently terminate at {canonical_dest} stations in dataset."
            )
            return {
                "status": "ERROR",
                "error": f"No railway fare records found for destination '{canonical_dest}' with travel class '{norm_class}'. {class_info}",
            }

        fare_per_person = int(round(matches["total_fare"].median()))
        total_cost = int(fare_per_person * num_travelers)
        median_dist = int(round(matches["distance_km"].median()))
        median_dur = int(round(matches["duration_mins"].median()))

        rep_dest_stn = str(matches["to_station_code"].mode().iloc[0])
        rep_dest_name = self.station_names.get(rep_dest_stn, rep_dest_stn)

        # Check if the destination is a gateway-only destination
        is_pure_gateway = (len(direct_stns) == 0 and len(gateway_stns) > 0)
        fare_source = "GATEWAY_DATASET_ESTIMATE" if is_pure_gateway else "DATASET_MEDIAN_ESTIMATE"
        gateway_note = (
            f"Direct railway connection to {canonical_dest} is unavailable; fares are estimated using railhead gateway {rep_dest_name}."
            if is_pure_gateway
            else None
        )
        gateway_disclaimer = (
            "Gateway estimate does not include the onward road/local transfer from the gateway station to the destination."
            if is_pure_gateway
            else None
        )

        return {
            "status": "SUCCESS",
            "destination": canonical_dest,
            "travel_class": norm_class,
            "origin_station": None,
            "origin_station_name": "NOT_SPECIFIED (Median aggregate across network routes)",
            "destination_station": rep_dest_stn,
            "destination_station_name": rep_dest_name,
            "is_gateway": is_pure_gateway,
            "gateway_station": rep_dest_stn if is_pure_gateway else None,
            "gateway_station_name": rep_dest_name if is_pure_gateway else None,
            "gateway_note": gateway_note,
            "gateway_disclaimer": gateway_disclaimer,
            "train_number": None,
            "train_name": f"{len(matches)} matching route-train combinations",
            "fare_per_person": fare_per_person,
            "travelers": num_travelers,
            "total_railway_cost": total_cost,
            "distance_km": median_dist,
            "duration_mins": median_dur,
            "fare_source": fare_source,
            "match_count": len(matches),
        }

    def _query_fare_records(
        self,
        dest_codes: List[str],
        norm_class: str,
        norm_origin: str,
        train_num: Optional[int],
    ) -> pd.DataFrame:
        """Helper to find and sort matching records deterministically."""
        mask = (
            (self.fares_df["to_station_code"].isin(dest_codes))
            & (self.fares_df["travel_class"] == norm_class)
            & (self.fares_df["from_station_code"] == norm_origin)
        )
        if train_num is not None:
            mask = mask & (self.fares_df["train_number"] == train_num)

        matched = self.fares_df[mask]
        if matched.empty:
            return matched

        return matched.sort_values(
            by=["distance_km", "total_fare", "train_number"], ascending=[True, True, True]
        )

    def _query_reverse_fare_records(
        self,
        dest_codes: List[str],
        norm_class: str,
        norm_origin: str,
        train_num: Optional[int],
    ) -> pd.DataFrame:
        """Helper to find matching records in the reverse direction (dest_codes -> norm_origin)."""
        mask = (
            (self.fares_df["from_station_code"].isin(dest_codes))
            & (self.fares_df["travel_class"] == norm_class)
            & (self.fares_df["to_station_code"] == norm_origin)
        )
        if train_num is not None:
            mask = mask & (self.fares_df["train_number"] == train_num)

        matched = self.fares_df[mask]
        if matched.empty:
            return matched

        return matched.sort_values(
            by=["distance_km", "total_fare", "train_number"], ascending=[True, True, True]
        )
