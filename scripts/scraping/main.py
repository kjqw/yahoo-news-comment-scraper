"""
実行前に`yahoo_news`データベースを作成しておく必要がある。
例えば、以下のコマンドで`yahoo_news`データベースを作成できる。
```sh
psql -h postgresql_db -U kjqw -d postgres -c "CREATE DATABASE yahoo_news;"
```
"""

# %%
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from db_manager import execute_query

# %%
# データベースの初期化
# %%
db_config = {
    "host": "postgresql_db",
    "database": "postgres",
    "user": "kjqw",
    "password": "1122",
    "port": "5432",
}
init_sql_path = Path(__file__).parents / "init.sql"
with init_sql_path.open() as f:
    query = f.read()
execute_query(query, db_config=db_config, commit=True)

# %%
