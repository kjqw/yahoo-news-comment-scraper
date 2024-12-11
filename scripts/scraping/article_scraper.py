# %%
import sys
from pathlib import Path

import functions
from classes import DBBase
from selenium.webdriver.common.by import By
from tqdm import tqdm
from xpaths.xpath_article_page import *

sys.path.append(str(Path(__file__).parents[1]))

from db_manager import execute_query


def get_and_save_articles(article_links: list[str], timeout: int = 10) -> None:
    """
    記事のリンクを入力して、記事の本文などの情報を取得する。
    """
    for article_link in tqdm(article_links):
        try:
            # ドライバを初期化
            driver = functions.init_driver(timeout)

            # ページを開く
            driver.get(article_link)

            article = DBBase()
            article.article_link = article_link

            article_section = driver.find_element(By.XPATH, XPATH_ARTICLE_SECTION)

            # 記事の要素を取得
            article.get_info(
                article_section,
                {
                    "article_title": RELATIVE_XPATH_ARTICLE_TITLE,
                    "author": RELATIVE_XPATH_ARTICLE_AUTHOR,
                    "posted_time": RELATIVE_XPATH_ARTICLE_POSTED_TIME,
                    "updated_time": RELATIVE_XPATH_ARTICLE_UPDATED_TIME,
                    "total_comment_count_with_reply": RELATIVE_XPATH_ARTICLE_COMMENT_COUNT,
                    "article_content": RELATIVE_XPATH_ARTICLE_CONTENT,
                    # "page_count": RELATIVE_XPATH_ARTICLE_PAGE_COUNT,
                    "learn_count": RELATIVE_XPATH_ARTICLE_LERAN_COUNT,
                    "clarity_count": RELATIVE_XPATH_ARTICLE_CLARITY_COUNT,
                    "new_perspective_count": RELATIVE_XPATH_ARTICLE_NEW_PERSPECTIVE_COUNT,
                },
            )

            # データベースに保存
            article.update_data("articles")

        except:
            pass

        finally:
            driver.quit()


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

    # 記事テーブルから記事のリンクを取得
    # Noneを除外してクエリを実行
    valid_article_ids = [str(data[1]) for data in article_data if data[1] is not None]

    if valid_article_ids:  # 有効なIDが存在する場合のみクエリを実行
        article_links = execute_query(
            f"""
            SELECT article_link
            FROM articles
            WHERE article_id IN ({','.join(valid_article_ids)})
            """
        )
        article_links = [link[0] for link in article_links]
    else:
        article_links = []  # 有効なIDがなければ空リストを返す

    print("Article Links:", article_links)

    # 記事のリンクを入力して、記事の本文などの情報を取得する。
    get_and_save_articles(article_links)

# %%
