from collections import defaultdict
from pathlib import Path

import utils
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from xpath import *


def get_reply_comment_sections(
    webelement: WebElement,
    max_replies: int,
) -> list[WebElement]:
    """
    返信コメントのセクションを取得する関数。

    Parameters
    ----------
    webelement : WebElement
        コメントセクションの親要素
    max_replies : int
        取得する返信の最大数

    Returns
    -------
    list[WebElement]
        返信コメントのセクションのリスト
    """
    # XPathの相対パスを取得
    relative_xpath_reply_count = utils.get_relative_xpath(
        XPATH_GENERAL_COMMENT_SECTIONS, XPATH_GENERAL_COMMENT_REPLY_COUNT
    )
    relative_xpath_reply_display_button = utils.get_relative_xpath(
        XPATH_GENERAL_COMMENT_SECTIONS, XPATH_GENERAL_COMMENT_REPLY_BUTTON
    )
    relative_xpath_reply_display_button_more = utils.get_relative_xpath(
        XPATH_GENERAL_COMMENT_SECTIONS, XPATH_GENERAL_COMMENT_REPLY_BUTTON_MORE
    )
    relative_xpath_reply_comment_sections = utils.get_relative_xpath(
        XPATH_GENERAL_COMMENT_SECTIONS, XPATH_REPLY_COMMENT_SECTIONS
    )

    # 返信の数を取得
    reply_count = int(
        webelement.find_elements(By.XPATH, relative_xpath_reply_count)[0].text
    )

    # 返信表示ボタンをクリック
    webelement.find_elements(By.XPATH, relative_xpath_reply_display_button)[0].click()

    reply_comments = []
    cnt = 10
    while (
        webelement.find_elements(By.XPATH, relative_xpath_reply_display_button_more)
        or cnt < max_replies
    ):
        # もっと返信を表示するボタンをクリック
        webelement.find_elements(By.XPATH, relative_xpath_reply_display_button_more)[
            0
        ].click()
        cnt += 10

    # 返信コメントのセクションを取得
    reply_comment_xpaths, reply_comments_sections = utils.find_all_combinations(
        webelement, relative_xpath_reply_comment_sections, [1, 1, max_replies]
    )
    return reply_comments_sections


def get_article_comments(
    url: str,
    max_comments: int,
    max_replies: int,
    order: str = "newer",
    timeout: int = 10,
) -> list[dict[str, str]]:
    """
    記事のコメントをスクレイピングする関数。

    Parameters
    ----------
    url : str
        記事のコメントページのURL
    max_comments : int
        取得するコメントの最大数
    max_replies : int
        取得する返信の最大数
    order : str, optional
        コメントの表示順。 "newer" または "recommended" のいずれかを指定, by default "newer"
    timeout : int, optional
        WebDriverのタイムアウト時間, by default 10

    Returns
    -------
    list[dict[str, str]]
        コメントのリスト
    """

    try:

        # ドライバを初期化
        driver = utils.init_driver(timeout)

        # 記事のコメントページを開く
        driver.get(url)

        # コメントセクションのリストを取得
        comment_sections = utils.get_comment_sections(driver)

        data = defaultdict(str)
        data["article_title"] = driver.find_element(By.XPATH, XPATH_ARTICLE_TITLE).text

        return data

        for comment_section in comment_sections:
            # 返信コメントのセクションを取得
            reply_comment_sections = get_reply_comment_sections(
                comment_section, max_replies
            )

    finally:
        driver.quit()


if __name__ == "__main__":
    url = "https://news.yahoo.co.jp/articles/ddae7ca185c389a92d2c1136a699a32fe4415094/comments"
    max_comments = 10
    max_replies = 5

    comments = get_article_comments(url, max_comments, max_replies)
    print(comments)
