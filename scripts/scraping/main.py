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
import article_scraper
import functions
import user_comment_scraper
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))

from db_manager import execute_query

# %%
# データベースの初期化
db_config = {
    "host": "postgresql_db",
    "database": "yahoo_news_restore",
    "user": "kjqw",
    "password": "1122",
    "port": "5432",
}
init_sql_path = Path(__file__).parent / "init.sql"
with init_sql_path.open() as f:
    query = f.read()
execute_query(query, db_config=db_config, commit=True)

# %%
max_comments_article_page = 30  # 記事ページをスクレイピングする際の最大コメント数
max_comments_user_page = 100  # ユーザーページをスクレイピングする際の最大コメント数
max_replies = 10  # 記事ページをスクレイピングする際の最大返信数
max_articles = 10  # スクレイピングする記事の最大数
max_users = 10  # スクレイピングするユーザーの最大数
timeout = 10  # webドライバのタイムアウト時間

# %%
# 現在ランキング上位の記事のリンクをスクレイピング
article_link_scraper.get_and_save_articles(db_config)

# %%
# まだ本文がスクレイピングされていない記事を取得
unprocessed_articles = execute_query(
    query=f"""
    SELECT article_id, article_link
    FROM articles
    WHERE article_content IS NULL
    ORDER BY scraped_time DESC;
    """,
    db_config=db_config,
)

# %%
# 記事の本文を取得
article_scraper.get_and_save_articles(
    db_config,
    [article_link for _, article_link in unprocessed_articles[:max_articles]],
)

# %%
# 記事ページからコメントを取得
unprocessed_article_comment_links = [
    (article_id, article_link + "/comments")
    for article_id, article_link in unprocessed_articles[:max_articles]
]
for article_id, article_comment_link in unprocessed_article_comment_links:
    article_comment_scraper.get_article_comments(
        db_config,
        article_id,
        article_comment_link,
        max_comments_article_page,
        max_replies,
        timeout=timeout,
    )

# %%
# usersテーブルにまだ記録されていないユーザーを取得
user_links = [
    i[0]
    for i in execute_query(
        query=f"""
        SELECT user_link
        FROM (
            SELECT DISTINCT ON (user_link) *
            FROM comments
        )
        WHERE user_link NOT IN (SELECT user_link FROM users)
        ORDER BY scraped_time DESC
        LIMIT {max_users};
        """,
        db_config=db_config,
    )
]

# 各ユーザーのユーザーページからコメントを取得
for user_link in tqdm(user_links):
    user_comment_scraper.get_and_save_articles_and_comments(
        db_config,
        user_link,
        max_comments_user_page,
    )

# %%
# 各ユーザーが見た記事の情報を取得
article_data = execute_query(
    f"""
    SELECT DISTINCT user_link, article_id
    FROM comments
    WHERE user_link IN ({','.join(f"'{i}'" for i in user_links)});
    """,
    db_config=db_config,
)
article_ids = [str(i[1]) for i in article_data if i[1] is not None]

# データベースからその記事のリンクを取得
article_links = execute_query(
    f"""
    SELECT article_link
    FROM articles
    WHERE article_id IN ({','.join(article_ids)});
    """,
    db_config=db_config,
)
article_links = [link[0] for link in article_links]

# %%
# まだ本文がスクレイピングされていない記事のリンクのみに絞る
unprocessed_article_links = execute_query(
    query=f"""
    SELECT article_id, article_link
    FROM articles
    WHERE article_content IS NULL AND article_link IN ({','.join(f"'{i}'" for i in article_links)})
    ORDER BY scraped_time DESC;
    """,
    db_config=db_config,
)

# %%
# 記事の本文を取得
article_scraper.get_and_save_articles(
    [article_link for _, article_link in unprocessed_article_links]
)

# %%
# posted_time の正規化
comment_ids, posted_times_comment, scraped_times_comment = zip(
    *execute_query(
        """
        SELECT comment_id, posted_time, scraped_time
        FROM comments
        """,
        db_config,
    ),
)
article_ids, posted_times_article, scraped_times_article = zip(
    *execute_query(
        """
        SELECT article_id, posted_time, scraped_time
        FROM articles
        """,
        db_config,
    ),
)

for comment_id, posted_times_comment, scraped_time_comment in zip(
    comment_ids, posted_times_comment, scraped_times_comment
):
    try:
        normalized_posted_time = functions.normalize_time(
            posted_times_comment, scraped_time_comment
        )
        # 正規化後のposted_timeを新たな列 normalized_posted_time として追加
        execute_query(
            f"""
            UPDATE comments
            SET normalized_posted_time = '{normalized_posted_time}'
            WHERE comment_id = '{comment_id}'
            """,
            db_config,
            commit=True,
        )
    except:
        pass

for article_id, posted_times_article, scraped_time_article in zip(
    article_ids, posted_times_article, scraped_times_article
):
    try:
        normalized_posted_time = functions.normalize_time(
            posted_times_article, scraped_time_article
        )
        # 正規化後のposted_timeを新たな列 normalized_posted_time として追加
        execute_query(
            f"""
            UPDATE articles
            SET normalized_posted_time = '{normalized_posted_time}'
            WHERE article_id = '{article_id}'
            """,
            db_config,
            commit=True,
        )
    except:
        pass
# %%
