import pandas as pd
import numpy as np

np.random.seed(42)

n_rows = 500

# True market value of each skin
true_value = np.random.uniform(50, 150, n_rows)

# Buy price: sometimes below true value (underpriced), sometimes above (overpriced)
# This models finding deals on the market — key to profitable trading
# Mix of fair-priced and underpriced skins (like sniping deals on the market)
# ~60% normal listings, ~40% underpriced deals
# Mix of fair-priced and underpriced skins (like sniping deals on the market)
# Mix of fair-priced and underpriced skins (like sniping deals on the market)
# Mix: ~55% are deals (sniped underpriced), ~45% are normal/overpriced listings
is_deal = np.random.random(n_rows) < 0.55
discount = np.where(is_deal,
                    np.random.uniform(-0.50, -0.18, n_rows),  # deals: 18-50% below true value
                    np.random.normal(0.05, 0.10, n_rows))      # normal: slightly overpriced on avg
buy_price = true_value * (1 + discount)
buy_price = np.clip(buy_price, 1, None)

price_ratio = buy_price / true_value  # <1 = underpriced, >1 = overpriced

num_listings = np.random.randint(1, 50, n_rows)
daily_volume = np.random.randint(1, 200, n_rows)
spread = np.random.uniform(0.5, 10, n_rows)
momentum = np.random.uniform(-1, 1, n_rows)

# Normalize features for the formula
daily_norm = daily_volume / (daily_volume.max() + 1e-6)
spread_norm = spread / (spread.max() + 1e-6)
listings_norm = num_listings / (num_listings.max() + 1e-6)

# Future price model:
# 1. Mean reversion: underpriced skins tend to rise back toward true value
# 2. Momentum: trending items continue (positive momentum = price going up)
# 3. Volume: high volume = more liquidity, slightly positive
# 4. Spread: high spread = unstable pricing, negative
# 5. Listings: many listings = supply pressure, negative
# 6. Noise: random market fluctuation

mean_reversion = (true_value - buy_price) / buy_price  # pull toward true value
trend = 0.18 * momentum                                 # momentum effect
volume_effect = 0.05 * daily_norm                        # liquidity bonus
spread_penalty = -0.03 * spread_norm                     # instability penalty
supply_pressure = -0.02 * listings_norm                  # supply penalty
noise = np.random.normal(0, 0.06, n_rows)               # market noise

# Blend: strong mean reversion (skins return toward true value over time) + other factors
# Skins revert strongly toward true value (Steam market is efficient over days/weeks)
pct_change = 0.92 * mean_reversion + trend + volume_effect + spread_penalty + supply_pressure + noise
future_price = buy_price * (1 + pct_change)

# Steam takes ~15% (13% Steam fee + 2% CS2 fee)
# You profit only if future_price > buy_price / 0.85
label = (future_price > buy_price / 0.85).astype(int)

df = pd.DataFrame({
    'price_ratio': price_ratio,
    'num_listings': num_listings,
    'daily_volume': daily_volume,
    'spread': spread,
    'momentum': momentum,
    'label': label
})

df.to_csv('fake_data.csv', index=False)

print(f"Profitable trades: {df['label'].mean() * 100:.2f}%")
print(f"Total rows: {len(df)}")
print(f"Profitable: {label.sum()}, Unprofitable: {(1-label).sum()}")
