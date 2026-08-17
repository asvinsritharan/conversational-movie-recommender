import duckdb

CATEGORY = "Movies_and_TV"
REVIEWS_PER_ITEM = 6

con = duckdb.connect()
con.execute("PRAGMA memory_limit='6GB'")

# For every item, keep the longest few 4-5 star reviews (substance + positive),
# writing item_id -> list of review texts as a compact Parquet.
con.execute(f"""
COPY (
    WITH ranked AS (
        SELECT asin AS item_id,
               reviewText,
               row_number() OVER (PARTITION BY asin
                                  ORDER BY length(reviewText) DESC) AS rn
        FROM read_json_auto('data/raw/{CATEGORY}.json',
                            format='newline_delimited', ignore_errors=true)
        WHERE reviewText IS NOT NULL
          AND length(reviewText) BETWEEN 100 AND 1200
          AND CAST(overall AS DOUBLE) >= 4.0
    )
    SELECT item_id, list(reviewText) AS reviews
    FROM ranked
    WHERE rn <= {REVIEWS_PER_ITEM}
    GROUP BY item_id
) TO 'data/review_store.parquet' (FORMAT parquet)
""")

n = con.sql("SELECT count(*) FROM 'data/review_store.parquet'").fetchone()[0]
print(f"[store] wrote reviews for {n:,} items -> data/review_store.parquet")