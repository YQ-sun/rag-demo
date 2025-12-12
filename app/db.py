#我要搭建自己的数据库
import lancedb 
import pyarrow as pa
import os

DB_PATH = "data/mydb"

def get_db():
    os.makedirs(DB_PATH, exist_ok=True)
    db =lancedb.connect(DB_PATH)

    if "documents" not in db.table_names():
        schema = pa.schema([
            pa.field("id", pa.int64(), nullable=False),
            pa.field("text", pa.string()),
            pa.field("embedding", lancedb.vector(1536))
        ])
        table = db.create_table("documents", schema=schema)
    else:
        table = db.open_table("documents")

    return table