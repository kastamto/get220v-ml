import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_URL = f"postgresql://{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASS','postgres')}@{os.getenv('DB_HOST','localhost')}:{os.getenv('DB_PORT','5432')}/{os.getenv('DB_NAME','thingsboard')}"

engine = create_engine(DB_URL)

def get_telemetry_raw(device_id: str, key: str, hours: int = 168):
    query = text(f"""
        SELECT t.ts,
               COALESCE(t.dbl_v, CAST(t.long_v AS double precision)) as value
        FROM ts_kv t
        JOIN key_dictionary k ON t.key = k.key_id
        WHERE t.entity_id = '{device_id}'
        AND k.key = '{key}'
        AND t.ts > (extract(epoch from now()) * 1000 - {hours * 3600 * 1000})
        ORDER BY t.ts ASC
        LIMIT 10000
    """)
    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()
        return [{"ts": r[0], "value": float(r[1]) if r[1] is not None else None} for r in rows]
