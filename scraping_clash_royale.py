from google_play_scraper import Sort, reviews
import pandas as pd
import numpy as np

# Konfigurasi
APP_ID = 'com.supercell.clashroyale' # ID Clash Royale di PlayStore
COUNT = 12000

def scrape_data():
    print(f"Mulai scraping {COUNT} ulasan untuk Clash Royale...")
    
    result, continuation_token = reviews(
        APP_ID,
        lang='en',
        country='id',
        sort=Sort.MOST_RELEVANT,
        count=COUNT,
        filter_score_with=None
    )

    df = pd.DataFrame(np.array(result).tolist())
    
    df = df[['content', 'score', 'at']]
    
    # Simpan ke CSV
    df.to_csv("clash_royale_reviews.csv", index=False)
    print(f"Berhasil menyimpan {len(df)} data ke clash_royale_reviews.csv")

if __name__ == "__main__":
    scrape_data()