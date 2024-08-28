from pathlib import Path

import utils
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement

XPATH_DICT = utils.read_xpath_json(Path(__file__).parent / "xpath.json")


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
    # XPathを読み込む
    xpath_comment_sections = XPATH_DICT["article_comment_page"]["general_comments"][
        "comment_sections"
    ]
    xpath_reply_count = XPATH_DICT["article_comment_page"]["general_comments"][
        "reply_count"
    ]
    xpath_reply_display_button = XPATH_DICT["article_comment_page"]["general_comments"][
        "reply_display_button"
    ]
    xpath_reply_display_button_more = XPATH_DICT["article_comment_page"][
        "general_comments"
    ]["reply_display_button_more"]
    xpath_reply_comment_sections = XPATH_DICT["article_comment_page"][
        "general_comments"
    ]["reply_comments"]["comment_sections"]

    # XPathの相対パスを取得
    relative_xpath_reply_count = utils.get_relative_xpath(
        xpath_comment_sections, xpath_reply_count
    )
    relative_xpath_reply_display_button = utils.get_relative_xpath(
        xpath_comment_sections, xpath_reply_display_button
    )
    relative_xpath_reply_display_button_more = utils.get_relative_xpath(
        xpath_comment_sections, xpath_reply_display_button_more
    )
    relative_xpath_reply_comment_sections = utils.get_relative_xpath(
        xpath_comment_sections, xpath_reply_comment_sections
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


def scrape_article_comments(
    url: str,
    max_comments: int,
    max_replies: int,
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

        return comment_sections

    finally:
        driver.quit()


if __name__ == "__main__":
    url = "https://news.yahoo.co.jp/articles/ddae7ca185c389a92d2c1136a699a32fe4415094/comments"
    max_comments = 10
    max_replies = 5

    comments = scrape_article_comments(url, max_comments, max_replies)
    print(comments)
