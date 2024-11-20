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


# %%
def get_and_save_articles_and_comments(
    db_config: dict,
    url: str,
    max_comments: int,
    timeout: int = 10,
):
    """
    ユーザーのページから、ユーザーがどの記事にどのようなコメントをしているかを取得する。

    Parameters
    ----------
    db_config : dict
        データベースの接続設定
    url : str
        ユーザーのページのURL
    timeout : int, Optional
        WebDriverのタイムアウト時間
    """

    try:
        # ドライバを初期化
        driver = functions.init_driver(timeout)

        # ページを開く
        driver.get(url)

        user = DBBase()

        # 基本情報を取得
        user.user_link = url
        user.get_info(
            driver,
            {
                "username": XPATH_USER_NAME,
                "total_comment_count": XPATH_TOTAL_COMMENT_COUNT,
                "total_agreements_count": XPATH_TOTAL_AGREEMENTS,
                "total_acknowledgements_count": XPATH_TOTAL_ACKNOWLEDGEMENTS,
                "total_disagreements_count": XPATH_TOTAL_DISAGREEMENTS,
            },
        )
        user.save_data("users", db_config)

        # 「もっと見る」ボタンが表示されるまで待機
        WebDriverWait(driver, timeout).until(
            EC.presence_of_all_elements_located((By.XPATH, XPATH_COMMENT_MORE_BUTTON))
        )
        cnt = 0
        tmp = len(driver.find_elements(By.XPATH, XPATH_COMMENT_SECTIONS))
        while cnt < max_comments // 10 and driver.find_elements(
            By.XPATH, XPATH_COMMENT_MORE_BUTTON
        ):

            # 「もっと見る」ボタンをクリック
            driver.find_element(By.XPATH, XPATH_COMMENT_MORE_BUTTON).click()

            # コメントセクションの数が増えているか確認
            WebDriverWait(driver, timeout).until(
                lambda driver: len(
                    driver.find_elements(By.XPATH, XPATH_COMMENT_SECTIONS)
                )
                > tmp
            )
            tmp = len(driver.find_elements(By.XPATH, XPATH_COMMENT_SECTIONS))

            cnt += 1
            print(f"{cnt}回目の「もっと見る」ボタンをクリックしました")

        # コメントセクションを取得
        WebDriverWait(driver, timeout).until(
            EC.presence_of_all_elements_located((By.XPATH, XPATH_COMMENT_SECTIONS))
        )
        comment_sections = driver.find_elements(By.XPATH, XPATH_COMMENT_SECTIONS)

        # ページの遷移を伴う処理を後でまとめて行うため、返信先と元のコメントのリンクを保存しておく
        reply_comment_links = []  # 返信先のコメントのリンク
        comment_ids = []  # 返信元のコメントのID

        # コメントセクションごとに処理
        for comment_section in comment_sections:
            for block in comment_section.find_elements(By.XPATH, "article"):
                # 記事
                if block.find_elements(By.XPATH, "a"):
                    article = DBBase()
                    article.get_info(
                        block,
                        {
                            "article_link": RELATIVE_XPATH_ARTICLE_LINK,
                            "article_title": RELATIVE_XPATH_ARTICLE_TITLE,
                            "author": RELATIVE_XPATH_ARTICLE_AUTHOR,
                            "posted_time": RELATIVE_XPATH_ARTICLE_POSTED_TIME,
                        },
                    )
                    article.save_data("articles", db_config)

                # 記事に対するコメント
                elif block.find_elements(By.XPATH, RELATIVE_XPATH_COMMENT_REPLY_COUNT):
                    comment = DBBase()
                    comment.username = user.username
                    comment.user_link = url
                    try:
                        comment.get_info(
                            block,
                            {
                                "posted_time": RELATIVE_XPATH_COMMENT_POSTED_TIME_1,
                                "comment_content": RELATIVE_XPATH_COMMENT_TEXT,
                                "agreements_count": RELATIVE_XPATH_COMMENT_AGREEMENTS,
                                "acknowledgements_count": RELATIVE_XPATH_COMMENT_ACKNOWLEDGEMENTS,
                                "disagreements_count": RELATIVE_XPATH_COMMENT_DISAGREEMENTS,
                                "reply_count": RELATIVE_XPATH_COMMENT_REPLY_COUNT,
                            },
                        )
                        comment.save_data("comments", db_config)

                    except:
                        pass

                # コメントに対する返信コメント
                elif block.find_elements(By.XPATH, RELATIVE_XPATH_COMMENT_TEXT):
                    comment = DBBase()
                    comment.username = user.username
                    comment.user_link = url
                    try:
                        comment.get_info(
                            block,
                            {
                                "posted_time": RELATIVE_XPATH_COMMENT_POSTED_TIME_2,
                                "comment_content": RELATIVE_XPATH_COMMENT_TEXT,
                                "agreements_count": RELATIVE_XPATH_COMMENT_AGREEMENTS,
                                "acknowledgements_count": RELATIVE_XPATH_COMMENT_ACKNOWLEDGEMENTS,
                                "disagreements_count": RELATIVE_XPATH_COMMENT_DISAGREEMENTS,
                            },
                        )
                        comment.save_data("comments", db_config)
                        comment_id = execute_query(
                            query=f"""
                            SELECT comment_id
                            FROM comments
                            WHERE user_link = '{url}'
                            AND comment_content = '{comment.comment_content}'
                            """,
                            db_config=db_config,
                        )[0][0]
                        comment_ids.append(comment_id)

                    except:
                        pass

                # 返信先のコメント
                elif block.find_elements(
                    By.XPATH, RELATIVE_XPATH_COMMENT_REPLY_COMMENT_LINK
                ):
                    reply_comment_link = block.find_element(
                        By.XPATH, RELATIVE_XPATH_COMMENT_REPLY_COMMENT_LINK
                    ).get_attribute("href")
                    reply_comment_links.append(reply_comment_link)

                # 削除されたコメント
                elif block.find_elements(
                    By.XPATH, RELATIVE_XPATH_COMMENT_REPLY_COMMENT_TEXT
                ):
                    print("削除されたコメント")

        for reply_comment_link, comment_id in zip(reply_comment_links, comment_ids):
            driver.get(reply_comment_link)
            element = driver.find_element(By.XPATH, XPATH_REPLY_COMMENT_SECTION)

            reply_comment = DBBase()
            reply_comment.get_info(
                element,
                {
                    "username": RELATIVE_XPATH_REPLY_COMMENT_USERNAME,
                    "user_link": RELATIVE_XPATH_REPLY_COMMENT_USERNAME,
                    "posted_time": RELATIVE_XPATH_REPLY_COMMENT_POSTED_TIME,
                    "comment_content": RELATIVE_XPATH_REPLY_COMMENT_TEXT,
                    "agreements_count": RELATIVE_XPATH_REPLY_COMMENT_AGREEMENTS,
                    "acknowledgements_count": RELATIVE_XPATH_REPLY_COMMENT_ACKNOWLEDGEMENTS,
                    "disagreements_count": RELATIVE_XPATH_REPLY_COMMENT_DISAGREEMENTS,
                },
            )
            reply_comment.save_data("comments", db_config)

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
    url = "https://news.yahoo.co.jp/users/eHCBfsjEFqRmy5UU44f-zPcdHpQ2j3hufgh5-VsS1Qx4Pg8600"
    max_comments = 20
    # max_comments = 10

    result = get_and_save_articles_and_comments(db_config, url, max_comments)
    print(result)
# %%
