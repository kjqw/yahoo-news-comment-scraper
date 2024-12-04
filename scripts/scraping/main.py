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
import user_comment_scraper
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))

from db_manager import execute_query

# %%
# データベースの初期化
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
default_max_comments = 30
default_max_replies = 10
default_max_articles = 10
default_timeout = 10

# %%
article_link_scraper.get_and_save_articles()

# %%
# 記事のリンクを取得
article_links = execute_query(
    query=f"""
    SELECT article_id, article_link, ranking
    FROM (
        SELECT article_id, article_link, ranking,
            ROW_NUMBER() OVER (PARTITION BY article_link ORDER BY ranking) AS rn
        FROM articles
    ) AS ranked_articles
    WHERE rn = 1
    ORDER BY ranking ASC
    LIMIT {default_max_articles};
    """,
    db_config=db_config,
)


# %%
# 記事ごとにコメントを取得
article_comment_links = [
    (article_id, article_link + "/comments")
    for article_id, article_link, ranking in article_links
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
user_links = [
    i[0]
    for i in execute_query(
        query=f"""
        SELECT user_link
        FROM comments
        WHERE agreements_count IS NOT NULL
        GROUP BY user_link
        ORDER BY MAX(agreements_count) DESC
        LIMIT 10;
        """,
        db_config=db_config,
    )
]
for user_link in tqdm(user_links):
    user_comment_scraper.get_and_save_articles_and_comments(
        db_config,
        user_link,
        default_max_comments,
    )

# %%
