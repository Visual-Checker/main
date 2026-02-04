import psycopg2

conn = psycopg2.connect(
    host="192.168.0.41",
    port=5432,
    dbname="VectorDB",
    user="orugu",
    password="orugu#0916",
)
cur = conn.cursor()

for table in ["face_recog_db", "hand_gesture_db", "voice_recog_db"]:
    cur.execute(
        f"SELECT vector_dims(embedding), COUNT(*) FROM {table} GROUP BY vector_dims(embedding) ORDER BY 1"
    )
    rows = cur.fetchall()
    if rows:
        print(f"{table}: {rows}")
    else:
        print(f"{table}: empty")

cur.close()
conn.close()
