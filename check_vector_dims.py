import psycopg2

conn = psycopg2.connect(
    host="192.168.0.41",
    port=5432,
    dbname="VectorDB",
    user="orugu",
    password="orugu#0916",
)
cur = conn.cursor()
cur.execute(
    "SELECT 'face_recog_db' AS table, COUNT(*), SUM(CASE WHEN vector_dims(embedding)=512 THEN 1 ELSE 0 END) FROM face_recog_db "
    "UNION ALL SELECT 'hand_gesture_db', COUNT(*), SUM(CASE WHEN vector_dims(embedding)=512 THEN 1 ELSE 0 END) FROM hand_gesture_db "
    "UNION ALL SELECT 'voice_recog_db', COUNT(*), SUM(CASE WHEN vector_dims(embedding)=512 THEN 1 ELSE 0 END) FROM voice_recog_db"
)
for table, total, dim512 in cur.fetchall():
    print(f"{table}: total={total}, dim512={dim512}")
cur.close()
conn.close()
