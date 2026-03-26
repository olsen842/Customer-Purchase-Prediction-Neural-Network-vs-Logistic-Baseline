import pandas as pd
import numpy as np

np.random.seed(42)

n_rows = 500

buy_price = np.random.uniform(50, 150, n_rows)
median_price = np.random.uniform(60, 140, n_rows)

noise = buy_price * np.random.normal(0, 0.05, n_rows)

num_listings = np.random.randint(1, 10, n_rows)
daily_volume = np.random.randint(1, 100, n_rows)
spread = np.random.uniform(0.5, 5, n_rows)
momentum = np.random.uniform(-1, 1, n_rows)

daily_normalized_volume = daily_volume / (daily_volume.max() + 1e-6)
spread_normalized = spread / (spread.max() + 1e-6)
listings_normalized = num_listings / (num_listings.max() + 1e-6)

price_ratio = buy_price / median_price 
future_price = buy_price * (1.10 + 0.2 * momentum 
                            + 0.1 * daily_normalized_volume 
                            - 0.02 * spread_normalized 
                            - 0.01 * listings_normalized) + noise


label = (future_price > buy_price / 0.85).astype(int)

df = pd.DataFrame({ 'price_ratio': price_ratio,
                   'num_listings': num_listings,
                   'daily_volume': daily_volume,
                   'spread': spread,
                   'momentum': momentum,
                   'label': label})

df.to_csv('fake_data.csv', index=False)

print(f"Profitable trades: {df['label'].mean() * 100:.2f}%")

#what i have fixed 
#1. I have fixed 0.30 to 0.10 in the future_price calculation to make it more realistic. if we speaking the 
# steam market wich is realtiv ly stable and not as volatile as the stock market, 
# a 30% change in price is quite high. A 10% change is more reasonable for the kind of price movements 
# we might expect in the Steam market. This adjustment should help create a more realistic dataset for training your model.
#2. i have added daily volume spread and num lsitings as features to the dataset. These features can provide

#fix 2
#1. i have added daily norm spread_norm and listings_norm to the future price calculation. 
# These features can help capture the impact of market conditions on price movements.