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

import article_comment_scraper
import article_link_scraper

sys.path.append(str(Path(__file__).parents[1]))

from db_manager import execute_query

# %%
# データベースの初期化
# %%
db_config = {
    "host": "postgresql_db",
    "database": "yahoo_news",
    "user": "kjqw",
    "password": "1122",
    "port": "5432",
}
init_sql_path = Path(__file__).parent / "init.sql"
with init_sql_path.open() as f:
    query = f.read()
execute_query(query, db_config=db_config, commit=True)

# %%
article_link_scraper.get_and_save_articles()
# %%
default_max_comments = 10
default_max_replies = 5
default_timeout = 10

# %%
# 記事のリンクを取得
article_links = execute_query(
    query="""
    SELECT article_id, article_link, ranking
    FROM (
        SELECT article_id, article_link, ranking,
            ROW_NUMBER() OVER (PARTITION BY article_link ORDER BY ranking) AS rn
        FROM articles
    ) AS ranked_articles
    WHERE rn = 1
    ORDER BY ranking ASC
    LIMIT 5
    """,
    db_config=db_config,
)


# %%
# 記事ごとにコメントを取得
article_comment_links = [
    (article_id, article_link + "/comments")
    for article_id, article_link, ranking in article_links[:2]
]
for article_id, article_comment_link in article_comment_links:
    article_comment_scraper.get_article_comments(
        db_config,
        article_id,
        article_comment_link,
        default_max_comments,
        default_max_replies,
        timeout=default_timeout,
    )

# %%
