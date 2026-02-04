import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="face_db",
    user="postgres",
    password="postgressql",
    port=5435
)
