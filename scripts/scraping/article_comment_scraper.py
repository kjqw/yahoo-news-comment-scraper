import sys
from datetime import datetime
from math import ceil
from pathlib import Path

import functions
from classes import DBBase
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
from xpaths.xpath_article_comment_page import *

sys.path.append(str(Path(__file__).parents[1]))

from db_manager import execute_query


def get_total_comment_count(driver: webdriver) -> list[str]:
    """
    コメント数を取得する関数。コメント要約AIがあるとコメント数のxpathが変わるため、関数として独立させた。

    Parameters
    ----------
    driver : webdriver
        記事のコメントぺージの1ページ目を開いたWebDriver

    Returns
    -------
    list[str]
        コメント数のリスト
    """

    def get_comment_count(xpaths: list[str]) -> str:
        """
        指定されたXPathリストから最初に見つけたコメント数を取得する。

        Parameters
        ----------
        xpaths : list[str]
            コメント数を取得するためのXPathリスト

        Returns
        -------
        str
            コメント数
        """
        for xpath in xpaths:
            try:
                return driver.find_element(By.XPATH, xpath).text
            except:
                continue
        return ""

    # コメント数をリストで取得
    total_comment_count_with_reply = get_comment_count(
        XPATHS_TOTAL_COMMENT_COUNT_WITH_REPLY
    )
    total_comment_count_without_reply = get_comment_count(
        XPATHS_TOTAL_COMMENT_COUNT_WITHOUT_REPLY
    )

    return [total_comment_count_with_reply, total_comment_count_without_reply]


def get_reply_comment_sections(
    webelement: WebElement,
    max_replies: int,
    timeout: int = 10,
) -> list[WebElement]:
    """
    返信コメントのセクションを取得する関数。

    Parameters
    ----------
    webelement : WebElement
        コメントセクションの親要素
    max_replies : int
        取得する返信の最大数
    timeout : int, Optional
        WebDriverのタイムアウト時間

    Returns
    -------
    list[WebElement]
        返信コメントのセクションのリスト
    """

    # 返信が0件の場合は空のリストを返す
    if (
        webelement.find_element(
            By.XPATH, RELATIVE_XPATH_GENERAL_COMMENT_REPLY_COUNT
        ).text
        == "0"
    ):
        return []

    # XPathの相対パスを取得
    relative_xpath_reply_comment_sections = functions.get_relative_xpath(
        XPATH_GENERAL_COMMENT_SECTIONS, XPATH_REPLY_COMMENT_SECTIONS
    )

    # 返信表示ボタンをクリック
    webelement.find_element(
        By.XPATH, RELATIVE_XPATH_GENERAL_COMMENT_REPLY_BUTTON
    ).click()

    # 返信コメントのセクションが表示されるまで待機
    WebDriverWait(webelement, timeout).until(
        EC.presence_of_all_elements_located(
            (By.XPATH, relative_xpath_reply_comment_sections)
        )
    )

    cnt = len(webelement.find_elements(By.XPATH, relative_xpath_reply_comment_sections))
    # 返信コメントの数がmax_repliesに達するか、もっと返信を表示するボタンがなくなるまで、コメントを追加で取得
    while (
        webelement.find_elements(
            By.XPATH, RELATIVE_XPATH_GENERAL_COMMENT_REPLY_BUTTON_MORE
        )
        and cnt < max_replies
    ):
        # もっと返信を表示するボタンをクリック
        webelement.find_element(
            By.XPATH, RELATIVE_XPATH_GENERAL_COMMENT_REPLY_BUTTON_MORE
        ).click()

        # 返信コメントのセクションが追加されるまで待機
        WebDriverWait(webelement, timeout).until(
            lambda element: len(
                element.find_elements(By.XPATH, relative_xpath_reply_comment_sections)
            )
            > cnt
        )

        # 更新後の返信コメントの数を取得
        cnt = len(
            webelement.find_elements(By.XPATH, relative_xpath_reply_comment_sections)
        )

    # 返信コメントのセクションを取得
    reply_comment_sections = webelement.find_elements(
        By.XPATH, relative_xpath_reply_comment_sections
    )
    return reply_comment_sections


def get_article_comments(
    db_config: dict[str, str],
    article_id: int,
    url: str,
    max_comments: int,
    max_replies: int,
    order: str = "recommended",
    timeout: int = 10,
):
    """
    記事のコメントをスクレイピングする関数。

    Parameters
    ----------
    article_id : int
        記事のID
    url : str
        記事のコメントページの1ページ目のURL
    max_comments : int
        取得するコメントの最大数。10の倍数以外の数を入力すると、10の倍数に切り上げて処理される。
    max_replies : int
        取得する返信の最大数。10の倍数以外の数を入力すると、10の倍数に切り上げて処理される。
    order : str, Optional
        コメントの表示順。 "newer" または "recommended" のいずれかを指定
    timeout : int, Optional
        WebDriverのタイムアウト時間
    db_config : dict[str, str], Optional
        データベースの接続情報
    """

    try:
        xpaths_general_comments = {
            "username": RELATIVE_XPATH_GENERAL_COMMENT_USERNAME,
            "user_link": RELATIVE_XPATH_GENERAL_COMMENT_USERNAME,
            "posted_time": RELATIVE_XPATH_GENERAL_COMMENT_POSTED_TIME,
            "comment_content": RELATIVE_XPATH_GENERAL_COMMENT_COMMENT_TEXT,
            "agreements_count": RELATIVE_XPATH_GENERAL_COMMENT_AGREEMENTS,
            "acknowledgements_count": RELATIVE_XPATH_GENERAL_COMMENT_ACKNOWLEDGEMENTS,
            "disagreements_count": RELATIVE_XPATH_GENERAL_COMMENT_DISAGREEMENTS,
            "reply_count": RELATIVE_XPATH_GENERAL_COMMENT_REPLY_COUNT,
        }
        xpaths_reply_comments = {
            "username": RELATIVE_XPATH_REPLY_COMMENT_USERNAME,
            "user_link": RELATIVE_XPATH_REPLY_COMMENT_USERNAME,
            "posted_time": RELATIVE_XPATH_REPLY_COMMENT_POSTED_TIME,
            "comment_content": RELATIVE_XPATH_REPLY_COMMENT_COMMENT_TEXT,
            "agreements_count": RELATIVE_XPATH_REPLY_COMMENT_AGREEMENTS,
            "acknowledgements_count": RELATIVE_XPATH_REPLY_COMMENT_ACKNOWLEDGEMENTS,
            "disagreements_count": RELATIVE_XPATH_REPLY_COMMENT_DISAGREEMENTS,
        }

        # ドライバを初期化
        driver = functions.init_driver(timeout)

        # 記事のコメントページを開く
        driver.get(url)

        # コメント数を取得
        total_comment_count_with_reply, total_comment_count_without_reply = (
            get_total_comment_count(driver)
        )
        total_comment_count_with_reply = functions.normalize_number(
            total_comment_count_with_reply
        )
        total_comment_count_without_reply = functions.normalize_number(
            total_comment_count_without_reply
        )

        execute_query(
            f"UPDATE articles SET total_comment_count_with_reply = {total_comment_count_with_reply} WHERE article_id = {article_id}",
            db_config,
            commit=True,
        )
        execute_query(
            f"UPDATE articles SET total_comment_count_without_reply = {total_comment_count_without_reply} WHERE article_id = {article_id}",
            db_config,
            commit=True,
        )

        # 最大数に達するまでコメントを取得
        page = 1
        while (page - 1) * 10 < max_comments:
            # 一般コメントのセクションを取得
            general_comment_sections = driver.find_elements(
                By.XPATH, XPATH_GENERAL_COMMENT_SECTIONS
            )
            # それぞれの一般コメントについて処理
            for general_comment_section in tqdm(
                general_comment_sections,
                desc=f"コメント取得 {page}/{ceil(max_comments//10)}ページ目",
            ):
                # 一般コメントのオブジェクトを作成
                general_comment = DBBase()

                # 一般コメントの情報を取得
                general_comment.get_info(
                    general_comment_section, xpaths_general_comments
                )

                general_comment.article_id = article_id

                # 数値を正規化
                general_comment.normalize_number()
                normalized_posted_time = functions.normalize_time(
                    general_comment.posted_time, datetime.now()
                )
                general_comment.normalized_posted_time = normalized_posted_time

                # 一般コメントの情報を保存
                general_comment.save_data("comments", db_config)

                # 一般コメントのIDをクラスに追加
                try:
                    # 同じ記事に対して同じユーザーが同じコメントを投稿している場合は片方のみを取得
                    general_comment.comment_id = execute_query(
                        f"""
                        SELECT comment_id FROM comments WHERE article_id = {article_id} AND user_link = '{general_comment.user_link}' AND comment_content = '{general_comment.comment_content}'
                        """,
                        db_config,
                    )[0][0]
                except:
                    continue

                # 返信コメントのセクションを取得
                reply_comment_sections = get_reply_comment_sections(
                    general_comment_section, max_replies
                )

                # 返信コメントがない場合はスキップ
                if reply_comment_sections:
                    # 返信コメントの情報を取得
                    for reply_comment_section in reply_comment_sections:
                        # 返信コメントのオブジェクトを作成
                        reply_comment = DBBase()

                        # 返信コメントの情報を取得
                        reply_comment.get_info(
                            reply_comment_section, xpaths_reply_comments
                        )

                        reply_comment.parent_comment_id = general_comment.comment_id
                        reply_comment.article_id = article_id

                        # 数値を正規化
                        reply_comment.normalize_number()
                        normalized_posted_time = functions.normalize_time(
                            reply_comment.posted_time, datetime.now()
                        )
                        reply_comment.normalized_posted_time = normalized_posted_time

                        # 返信コメントの情報を保存
                        reply_comment.save_data("comments", db_config)

            # 次のページに移動
            page += 1
            functions.open_page(driver, url, page, order)

    finally:
        driver.quit()


if __name__ == "__main__":
    default_max_comments = 20
    default_max_replies = 20
    # default_order = "recommended"
    default_timeout = 10
    db_config = {
        "host": "postgresql_db",
        "database": "yahoo_news",
        "user": "kjqw",
        "password": "1122",
        "port": "5432",
    }

    # 記事のリンクを取得
    # article_links = execute_query(
    #     """
    #     SELECT article_id, article_link, ranking
    #     FROM (
    #         SELECT article_id, article_link, ranking,
    #             ROW_NUMBER() OVER (PARTITION BY article_link ORDER BY ranking) AS rn
    #         FROM articles
    #     ) AS ranked_articles
    #     WHERE rn = 1
    #     ORDER BY ranking ASC
    #     LIMIT 5
    #     """
    # )
    article_links = execute_query(
        """
        SELECT article_id, article_link, comment_count_per_hour
        FROM articles
        WHERE comment_count_per_hour IS NOT NULL
        ORDER BY comment_count_per_hour desc
        LIMIT 5
        """
    )

    # print(article_links)

    # 記事ごとにコメントを取得
    article_comment_links = [
        (article_id, article_link + "/comments")
        for article_id, article_link, _ in article_links
    ]
    for article_id, article_comment_link in article_comment_links:
        get_article_comments(
            db_config,
            article_id,
            article_comment_link,
            default_max_comments,
            default_max_replies,
            timeout=default_timeout,
        )
