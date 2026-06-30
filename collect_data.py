import requests
import csv
import time
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# --------------- CONFIGURATION ---------------

# How often to poll (in seconds). 4 hours = 14400
POLL_INTERVAL = 14400

# Delay between individual requests to avoid rate limiting (seconds)
REQUEST_DELAY = 5

# Where to save the data
CSV_FILE = "market_data.csv"

# File to store last known prices for momentum calculation
STATE_FILE = "collector_state.json"

# Currency (1=USD, 3=EUR, 6=GBP, 9=NOK, 18=DKK)
CURRENCY = 3

#50-80 skins is a comfortable sweet spot
#har 20 nu
SKINS = [
    "AK-47 | Redline (Field-Tested)",
    "AWP | Asiimov (Field-Tested)",
    "M4A1-S | Hyper Beast (Field-Tested)",
    "USP-S | Kill Confirmed (Field-Tested)",
    "Glock-18 | Fade (Factory New)",
    "AK-47 | Vulcan (Field-Tested)",
    "AWP | Lightning Strike (Factory New)",
    "M4A4 | Desolate Space (Field-Tested)",
    "Desert Eagle | Blaze (Factory New)",
    "AK-47 | Neon Rider (Field-Tested)",
    "AWP | Hyper Beast (Field-Tested)",
    "M4A1-S | Golden Coil (Field-Tested)",
    "AWP | Containment Breach (Field-Tested)",
    "AK-47 | Ice Coaled (Field-Tested)",
    "M4A4 | The Emperor (Field-Tested)",
    "USP-S | Printstream (Field-Tested)",
    "Glock-18 | Gamma Doppler (Factory New)",
    "AK-47 | Phantom Disruptor (Field-Tested)",
    "AWP | Chromatic Aberration (Field-Tested)",
    "M4A1-S | Printstream (Field-Tested)",
]

# -------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("collector.log"),
    ],
)
log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})


def load_state() -> dict:
    """Load previous prices for momentum calculation."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def parse_price(price_str: str) -> float | None:
    """Parse Steam price strings like '$12.34' or '12,34€' into float."""
    if not price_str:
        return None
    cleaned = price_str.replace("$", "").replace("€", "").replace("£", "")
    cleaned = cleaned.replace("kr", "").replace(",", ".").replace(" ", "").strip()
    # Handle cases like "1.234.56" (thousands separator)
    parts = cleaned.split(".")
    if len(parts) > 2:
        cleaned = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return float(cleaned)
    except ValueError:
        return None


def get_price_overview(market_hash_name: str) -> dict | None:
    """
    Returns: { lowest_price, median_price, volume }
    Uses: /market/priceoverview/
    """
    url = "https://steamcommunity.com/market/priceoverview/"
    params = {
        "appid": 730,
        "currency": CURRENCY,
        "market_hash_name": market_hash_name,
    }
    try:
        resp = SESSION.get(url, params=params, timeout=15)
        if resp.status_code == 429:
            log.warning("Rate limited on priceoverview. Sleeping 60s...")
            time.sleep(60)
            return None
        if resp.status_code != 200:
            log.warning(f"priceoverview returned {resp.status_code} for {market_hash_name}")
            return None
        data = resp.json()
        if not data.get("success"):
            return None
        return data
    except Exception as e:
        log.error(f"Error fetching priceoverview for {market_hash_name}: {e}")
        return None


def get_listings_count(market_hash_name: str) -> int | None:
    """
    Returns total number of active listings for this item.
    Uses: /market/listings/730/{name}/render/
    """
    url = f"https://steamcommunity.com/market/listings/730/{requests.utils.quote(market_hash_name)}/render/"
    params = {
        "start": 0,
        "count": 1,
        "currency": CURRENCY,
        "norender": 1,
    }
    try:
        resp = SESSION.get(url, params=params, timeout=15)
        if resp.status_code == 429:
            log.warning("Rate limited on listings. Sleeping 60s...")
            time.sleep(60)
            return None
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("total_count")
    except Exception as e:
        log.error(f"Error fetching listings for {market_hash_name}: {e}")
        return None


def collect_one_skin(skin: str, state: dict) -> dict | None:
    """Collect all 5 features for one skin."""

    # --- price overview ---
    overview = get_price_overview(skin)
    if not overview:
        return None
    time.sleep(REQUEST_DELAY)

    lowest = parse_price(overview.get("lowest_price", ""))
    median = parse_price(overview.get("median_price", ""))
    volume_str = overview.get("volume", "0")
    daily_volume = int(volume_str.replace(",", "")) if volume_str else 0

    if not lowest or not median or median == 0:
        log.warning(f"Bad price data for {skin}: lowest={lowest}, median={median}")
        return None

    # --- listings count ---
    num_listings = get_listings_count(skin)
    if num_listings is None:
        num_listings = 0
    time.sleep(REQUEST_DELAY)

    # --- computed features ---

    # 1. price_ratio = lowest_price / median_price
    price_ratio = round(lowest / median, 4)

    # 2. spread = absolute difference between lowest listing and median
    spread = round(abs(lowest - median), 4)

    # 3. momentum = price change since last poll
    prev_price = state.get(skin, {}).get("last_price")
    if prev_price and prev_price > 0:
        momentum = round((lowest - prev_price) / prev_price, 4)
    else:
        momentum = 0.0  # first reading, no momentum yet

    # Update state for next poll
    state[skin] = {"last_price": lowest}

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "skin": skin,
        "price_ratio": price_ratio,
        "num_listings": num_listings,
        "daily_volume": daily_volume,
        "spread": spread,
        "momentum": momentum,
        "lowest_price": lowest,
        "median_price": median,
    }


def init_csv():
    """Create CSV with headers if it doesn't exist."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "skin",
                "price_ratio", "num_listings", "daily_volume", "spread", "momentum",
                "lowest_price", "median_price",
            ])
        log.info(f"Created {CSV_FILE}")


def append_row(row: dict):
    """Append one row to the CSV."""
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["timestamp"], row["skin"],
            row["price_ratio"], row["num_listings"], row["daily_volume"],
            row["spread"], row["momentum"],
            row["lowest_price"], row["median_price"],
        ])


def run_one_cycle(state: dict) -> dict:
    """Poll all skins once."""
    log.info(f"--- Starting collection cycle for {len(SKINS)} skins ---")
    collected = 0

    for skin in SKINS:
        log.info(f"Collecting: {skin}")
        row = collect_one_skin(skin, state)
        if row:
            append_row(row)
            collected += 1
            log.info(
                f"  OK: ratio={row['price_ratio']} listings={row['num_listings']} "
                f"vol={row['daily_volume']} spread={row['spread']} mom={row['momentum']}"
            )
        else:
            log.warning(f"  SKIP: failed to collect {skin}")

    save_state(state)
    log.info(f"--- Cycle done: {collected}/{len(SKINS)} skins collected ---")
    return state


def main():
    log.info("=" * 50)
    log.info("CS2 Market Data Collector starting")
    log.info(f"Tracking {len(SKINS)} skins every {POLL_INTERVAL // 3600}h")
    log.info(f"Saving to {CSV_FILE}")
    log.info("=" * 50)

    init_csv()
    state = load_state()

    while True:
        try:
            state = run_one_cycle(state)
        except Exception as e:
            log.error(f"Cycle failed: {e}", exc_info=True)

        next_run = datetime.utcnow().isoformat()
        log.info(f"Sleeping {POLL_INTERVAL}s until next cycle...")

        try:
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            log.info("Stopped by user. Data saved.")
            break


if __name__ == "__main__":
    main()
