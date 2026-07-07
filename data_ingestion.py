import duckdb

# --- knobs (all in one place so they're easy to see) ---
CATEGORY = "Movies_and_TV"
MIN_USER_INTERACTIONS = 5   # 5-core: drop users with fewer than 5 reviews
MIN_ITEM_INTERACTIONS = 5   # 5-core: drop items with fewer than 5 reviews

reviews_path = f"data/raw/{CATEGORY}.json"
meta_path    = f"data/raw/meta_{CATEGORY}.json"

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='6GB'")

con.execute(f"""
CREATE VIEW raw_rev AS
SELECT reviewerID                  AS user_id,
       asin                        AS item_id,
       CAST(overall AS DOUBLE)     AS rating,
       CAST(unixReviewTime AS BIGINT) AS ts
FROM read_json_auto('{reviews_path}', format='newline_delimited', ignore_errors=true)
WHERE reviewerID IS NOT NULL AND asin IS NOT NULL
""")

con.execute(f"""
CREATE TABLE inter AS
WITH keep_items AS (
    SELECT item_id FROM raw_rev
    GROUP BY item_id HAVING count(*) >= {MIN_ITEM_INTERACTIONS}
),
filtered AS (
    SELECT * FROM raw_rev WHERE item_id IN (SELECT item_id FROM keep_items)
),
keep_users AS (
    SELECT user_id FROM filtered
    GROUP BY user_id HAVING count(*) >= {MIN_USER_INTERACTIONS}
)
SELECT * FROM filtered WHERE user_id IN (SELECT user_id FROM keep_users)
""")
con.execute("COPY (SELECT * FROM inter) TO 'data/interactions.parquet' (FORMAT parquet)")

con.execute(f"""
CREATE VIEW raw_meta AS
SELECT asin AS item_id,
       COALESCE(title, '') AS title,
       array_to_string(COALESCE(description, []), ' ') AS descr
FROM read_json_auto('{meta_path}', format='newline_delimited', ignore_errors=true)
WHERE asin IS NOT NULL
""")
con.execute("""
COPY (
    SELECT m.item_id, m.title, trim(m.title || '. ' || m.descr) AS text
    FROM raw_meta m
    JOIN (SELECT DISTINCT item_id FROM inter) keep USING (item_id)
) TO 'data/items.parquet' (FORMAT parquet)
""")

n  = con.sql("SELECT count(*) FROM inter").fetchone()[0]
u  = con.sql("SELECT count(DISTINCT user_id) FROM inter").fetchone()[0]
it = con.sql("SELECT count(DISTINCT item_id) FROM inter").fetchone()[0]
print(f"[ingest] interactions={n:,}  users={u:,}  items={it:,}")
print("[ingest] wrote data/interactions.parquet and data/items.parquet")