"""
ユーザーページからコメントを取得してモデリングのためのデータ集めをするためのスクリプト。
既存のデータベースに追加する処理は面倒なので、commentsテーブルのみを空にした新しいデータベースを作成し、そこにスクレイピング結果を保存する。
"""

# %%
import sys
from pathlib import Path

import article_scraper
import user_comment_scraper
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))

from db_manager import execute_query

# %%
# データベースの初期化
db_config = {
    "host": "postgresql_db",
    "database": "yahoo_news_modeling_1",
    "user": "kjqw",
    "password": "1122",
    "port": "5432",
}
init_sql_path = Path(__file__).parent / "init.sql"
with init_sql_path.open() as f:
    query = f.read()
execute_query(query, db_config=db_config, commit=True)

# %%
max_comments_user_page = 1000  # ユーザーページをスクレイピングする際の最大コメント数
max_users = 10  # スクレイピングするユーザーの最大数
min_total_comment_count = 100  # スクレイピングするユーザーの最小コメント数
timeout = 10  # webドライバのタイムアウト時間

# %%
# usersテーブルに記録されているユーザーを、total_comment_countが多い順に取得
user_links = [
    i[0]
    for i in execute_query(
        query=f"""
        SELECT user_link
        FROM users
        WHERE total_comment_count >= {min_total_comment_count}
        ORDER BY total_comment_count DESC
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
# user_linkからuser_idを特定して、commentsテーブルにuser_idを追加
user_id_links = execute_query(
    """
    SELECT user_id, user_link
    FROM users;
    """,
    db_config=db_config,
)
for user_id, user_link in user_id_links:
    execute_query(
        f"""
        UPDATE comments
        SET user_id = '{user_id}'
        WHERE user_link = '{user_link}';
        """,
        db_config=db_config,
        commit=True,
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
    db_config, [article_link for _, article_link in unprocessed_article_links]
)

# %%
