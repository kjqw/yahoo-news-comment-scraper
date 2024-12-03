# %%
import sys
from pathlib import Path

import functions
from classes import DBBase
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from xpaths.xpath_user_page import *

sys.path.append(str(Path(__file__).parents[1]))

from db_manager import execute_query

if __name__ == "__main__":
    db_config = {
        "host": "postgresql_db",
        "database": "yahoo_news",
        "user": "kjqw",
        "password": "1122",
        "port": "5432",
    }
    max_users = 10  # 投稿数が多い上位Nユーザーを取得

    # 投稿コメント数の多いユーザーを取得
    user_links = execute_query(
        f"""
        SELECT user_link
        FROM users
        WHERE total_comment_count IS NOT NULL
        ORDER BY total_comment_count DESC
        LIMIT {max_users}
        """,
    )
    print("User Links:", user_links)

    # 各ユーザーが見た全ての記事のリンクを取得
    user_links_list = [f"'{link[0]}'" for link in user_links]
    article_data = execute_query(
        f"""
        SELECT user_link, article_id
        FROM comments
        WHERE user_link IN ({','.join(user_links_list)})
        GROUP BY user_link, article_id
        ORDER BY user_link
        """
    )
    print("Raw Article Data:", article_data)

    # 辞書形式に整形
    user_article_dict = {}
    for user_link, article_id in article_data:
        if user_link not in user_article_dict:
            user_article_dict[user_link] = []
        user_article_dict[user_link].append(article_id)

    print("User to Articles Dictionary:", user_article_dict)

    # article_links = execute_query(


# %%
n = 0
print(len(user_article_dict[user_links[n][0]]))
print(user_article_dict[user_links[n][0]])
# %%
