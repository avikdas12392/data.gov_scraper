import os
import time
import csv
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
import pandas as pd
from difflib import SequenceMatcher

# ---------------------------
# Configuration (edit below)
# ---------------------------
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "016aadea6e666db587bcfd18254b01ab7a238fa3")
FOLDER_PATH = r"D:\Study - Gufic\Python\serper_project_private circle"
INPUT_FILE = "input_pc1.csv"   # Input must have: ogno, ogtype, ogname, ogcity, ogstate, ogpin, ogadd
OUTPUT_FILE = "output_enriched.csv"

SERPER_PLACES_URL = "https://google.serper.dev/places"

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5
DELAY_BETWEEN_CALLS = 1.0

NAME_WEIGHT = 0.7
CITY_WEIGHT = 0.3

EXPECTED_INPUT_COLUMNS = ['ogno', 'ogtype', 'ogname', 'ogcity', 'ogstate', 'ogpin', 'ogadd']

APPENDED_COLUMNS = [
    "Name", "Address", "City", "Pincode", "Keyword",
    "Latitude", "Longitude", "Rating", "Number of Reviews",
    "Category", "Phone", "Website", "CID", "Google Maps URL",
    "Match Confidence"
]

PIN_PATTERN = re.compile(r'\b\d{6}\b')

# ---------------------------
# Helpers
# ---------------------------
def safe_str(x: Any) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)

def seq_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r'\s+', ' ', s.strip())

def extract_pincode(address: str) -> str:
    if not address:
        return ""
    m = PIN_PATTERN.search(address)
    return m.group(0) if m else ""

def extract_city_from_address(address: str) -> str:
    if not address:
        return ""
    parts = [p.strip() for p in address.split(',') if p.strip()]
    if len(parts) >= 2:
        return parts[-2]
    return parts[0] if parts else ""

def float_or_blank(value: Any) -> Any:
    try:
        if value is None or value == "":
            return ""
        return float(value)
    except Exception:
        return ""

def int_or_blank(value: Any) -> Any:
    try:
        if value is None or value == "":
            return ""
        return int(float(value))
    except Exception:
        return ""

# ---------------------------
# Serper wrapper
# ---------------------------
def serper_places_search(query: str, api_key: str) -> Optional[Dict[str, Any]]:
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = {"q": query, "gl": "in", "hl": "en"}
    backoff = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(SERPER_PLACES_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            status = getattr(e.response, "status_code", None) if hasattr(e, 'response') else None
            if status and 400 <= status < 500 and status != 429:
                print(f"[serper] Non-retriable error {status} for '{query}': {e}")
                return None
            if attempt < MAX_RETRIES:
                print(f"[serper] Attempt {attempt}/{MAX_RETRIES} failed for '{query}': {e}. Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= RETRY_BACKOFF
            else:
                print(f"[serper] All {MAX_RETRIES} attempts failed for '{query}'.")
    return None

# ---------------------------
# Matching & mapping
# ---------------------------
def pick_best_place(places: List[Dict[str, Any]], original_name: str, original_city: str) -> Tuple[Optional[Dict[str, Any]], float]:
    if not places:
        return None, 0.0
    orig_name = normalize_text(original_name)
    orig_city = normalize_text(original_city)
    best_score = -1.0
    best_place = None
    for p in places:
        title = normalize_text(p.get("title") or p.get("name") or "")
        address_city = normalize_text(p.get("address") or p.get("description") or "")
        candidate_city = normalize_text(p.get("city") or extract_city_from_address(address_city))
        name_score = seq_ratio(orig_name, title)
        city_score = seq_ratio(orig_city, candidate_city)
        combined = (NAME_WEIGHT * name_score) + (CITY_WEIGHT * city_score)
        lat = p.get("latitude") or p.get("lat")
        lon = p.get("longitude") or p.get("lng") or p.get("lon")
        has_geo = 1.0 if (lat is not None and lon is not None and str(lat) != "" and str(lon) != "") else 0.0
        combined += 0.02 * has_geo
        if combined > best_score:
            best_score = combined
            best_place = p
    return best_place, best_score

def map_place_to_output(place: Dict[str, Any], keyword: str, match_confidence: float) -> Dict[str, Any]:
    if not place:
        return {col: "" for col in APPENDED_COLUMNS}
    title = safe_str(place.get("title") or place.get("name") or "")
    address = safe_str(place.get("address") or place.get("description") or place.get("formatted_address") or "")
    city = safe_str(place.get("city") or extract_city_from_address(address) or "")
    pincode = safe_str(place.get("postalCode") or extract_pincode(address) or "")

    lat = place.get("latitude") or place.get("lat") or ""
    lon = place.get("longitude") or place.get("lng") or place.get("lon") or ""

    rating = place.get("rating") if "rating" in place else ""
    rating_count = place.get("ratingCount") or place.get("reviews") or place.get("reviewCount") or ""

    category = safe_str(place.get("category") or place.get("type") or "")
    phone = safe_str(place.get("phoneNumber") or place.get("phone") or place.get("contact") or "")
    website = safe_str(place.get("website") or place.get("url") or "")

    # ✅ EXACT cid only if Serper returned 'cid'
    cid = safe_str(place.get("cid")) if "cid" in place and place.get("cid") not in (None, "") else ""

    # ✅ Only use a maps URL if returned directly by Serper (no construction)
    maps_url = ""
    if "mapsUrl" in place and place.get("mapsUrl"):
        maps_url = safe_str(place.get("mapsUrl"))
    elif "maps_url" in place and place.get("maps_url"):
        maps_url = safe_str(place.get("maps_url"))

    lat = float_or_blank(lat)
    lon = float_or_blank(lon)
    rating = float_or_blank(rating)
    rating_count = int_or_blank(rating_count)

    return {
        "Name": title,
        "Address": address,
        "City": city,
        "Pincode": pincode,
        "Keyword": keyword,
        "Latitude": lat,
        "Longitude": lon,
        "Rating": rating,
        "Number of Reviews": rating_count,
        "Category": category,
        "Phone": phone,
        "Website": website,
        "CID": cid,
        "Google Maps URL": maps_url,
        "Match Confidence": float(match_confidence) if match_confidence is not None else ""
    }

# ---------------------------
# Main enrichment
# ---------------------------
def enrich_directory(input_path: Path, output_path: Path, api_key: str):
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])
    print(f"Loaded input file with {len(df)} rows.")

    missing = [c for c in EXPECTED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing expected columns: {missing}")

    out_rows = []
    processed_count = 0
    skipped_count = 0

    for idx, row in df.iterrows():
        ogno = safe_str(row.get("ogno", "")).strip()
        ogtype = safe_str(row.get("ogtype", "")).strip()
        ogname = safe_str(row.get("ogname", "")).strip()
        ogcity = safe_str(row.get("ogcity", "")).strip()
        ogstate = safe_str(row.get("ogstate", "")).strip()
        ogpin = safe_str(row.get("ogpin", "")).strip()
        ogadd = safe_str(row.get("ogadd", "")).strip()

        if not ogname or not ogcity:
            skipped_count += 1
            print(f"[{idx+1}] Skipping (missing ogname/ogcity). ogname='{ogname}', ogcity='{ogcity}'")
            continue

        processed_count += 1
        search_keyword = f"{ogname} {ogcity}"
        print(f"[{idx+1}] Searching: {search_keyword}")

        api_result = serper_places_search(search_keyword, api_key)
        places = []
        if api_result and isinstance(api_result, dict):
            if "places" in api_result and isinstance(api_result["places"], list):
                places = api_result["places"]
            else:
                for k in ("local_results", "results", "organic", "items"):
                    if k in api_result and isinstance(api_result[k], list):
                        places = api_result[k]
                        break

        if not places:
            print(f"[{idx+1}] No places returned; leaving appended fields blank.")
            mapped = {col: "" for col in APPENDED_COLUMNS}
            mapped["Keyword"] = search_keyword
            mapped["Match Confidence"] = ""
        else:
            best_place, best_score = pick_best_place(places, ogname, ogcity)
            if best_place is None:
                best_place = places[0]
                best_score = 0.0
            print(f"[{idx+1}] Best match score={best_score:.4f}. Title='{best_place.get('title') or best_place.get('name') or ''}'")
            mapped = map_place_to_output(best_place, search_keyword, best_score)

        out_row = {
            "ogno": ogno,
            "ogtype": ogtype,
            "ogname": ogname,
            "ogcity": ogcity,
            "ogstate": ogstate,
            "ogpin": ogpin,
            "ogadd": ogadd
        }
        out_row.update(mapped)
        out_rows.append(out_row)

        time.sleep(DELAY_BETWEEN_CALLS)

    output_columns = EXPECTED_INPUT_COLUMNS + APPENDED_COLUMNS
    df_out = pd.DataFrame(out_rows, columns=output_columns)

    # Ensure numeric columns are floats/ints or blank
    for col in ("Latitude", "Longitude", "Rating"):
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(lambda v: float(v) if (v != "" and v is not None and str(v).strip() != "") else "")
    if "Number of Reviews" in df_out.columns:
        df_out["Number of Reviews"] = df_out["Number of Reviews"].apply(lambda v: int(v) if (v != "" and v is not None and str(v).strip() != "") else "")
    if "Match Confidence" in df_out.columns:
        df_out["Match Confidence"] = df_out["Match Confidence"].apply(lambda v: float(v) if (v != "" and v is not None and str(v).strip() != "") else "")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")

    print(f"\nDone. Processed {processed_count} rows, skipped {skipped_count} rows.")
    print(f"Output saved to: {output_path} ({len(df_out)} rows)")

# ---------------------------
# Script entry
# ---------------------------
if __name__ == "__main__":
    try:
        input_path = Path(FOLDER_PATH) / INPUT_FILE
        output_path = Path(FOLDER_PATH) / OUTPUT_FILE

        if not SERPER_API_KEY:
            raise RuntimeError("No Serper API key provided. Set SERPER_API_KEY env var or update the script.")

        print("="*60)
        print("Business directory enrichment (Serper Places)")
        print("="*60)
        print(f"Input : {input_path}")
        print(f"Output: {output_path}\n")

        enrich_directory(input_path, output_path, SERPER_API_KEY)

        print("\nAll done.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        raise