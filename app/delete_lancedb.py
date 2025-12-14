from lancedb import connect
import lancedb
import pyarrow as pa

db = connect("/app/data/mydb")  # 或者内存模式
# 删除 table
if "documents" in db.table_names():
    db.drop_table("documents")
    print("Table 'documents' dropped!")

# 重新创建空 table
schema = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("text", pa.string()),
    pa.field("embedding", lancedb.vector(1536))
])
table = db.create_table("documents", schema=schema)
print("Empty table 'documents' created!")
