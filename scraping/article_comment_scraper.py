import pickle
from collections import defaultdict
from pathlib import Path

import utils
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from xpath import *


class GeneralComment:
    """
    一般コメントの情報を格納するクラス。

    Attributes
    ----------
    username : str
        ユーザ名
    posted_time : str
        投稿日時
    comment_text : str
        コメントの本文
    agreements : int
        「共感した」の数
    acknowledgements : int
        「参考になった」の数
    disagreements : int
        「うーん」の数
    reply_count : int
        返信の数
    reply_comments : list[ReplyComment]
        返信コメントのリスト
    """

    def __init__(
        self,
        username: str,
        posted_time: str,
        comment_text: str,
        agreements: int,
        acknowledgements: int,
        disagreements: int,
        reply_count: int,
    ):
        # コメントの情報
        self.username: str = username
        self.posted_time: str = posted_time
        self.comment_text: str = comment_text
        self.agreements: int = agreements
        self.acknowledgements: int = acknowledgements
        self.disagreements: int = disagreements
        self.reply_count: int = reply_count

        # 返信コメントのリスト
        self.reply_comments: list[ReplyComment] = []


class ReplyComment:
    """
    返信コメントの情報を格納するクラス。

    Attributes
    ----------
    username : str
        ユーザ名
    posted_time : str
        投稿日時
    comment_text : str
        コメントの本文
    agreements : int
        「共感した」の数
    acknowledgements : int
        「参考になった」の数
    disagreements : int
        「うーん」の数
    base_comment : GeneralComment | None
        返信先の一般コメント
    """

    def __init__(
        self,
        username: str,
        posted_time: str,
        comment_text: str,
        agreements: int,
        acknowledgements: int,
        disagreements: int,
    ):
        # コメントの情報
        self.username: str = username
        self.posted_time: str = posted_time
        self.comment_text: str = comment_text
        self.agreements: int = agreements
        self.acknowledgements: int = acknowledgements
        self.disagreements: int = disagreements

        # 返信先のコメント
        self.base_comment: GeneralComment | None = None

    def set_base_comment(self, base_comment: GeneralComment) -> None:
        self.base_comment = base_comment


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
    relative_xpath_reply_comment_sections = utils.get_relative_xpath(
        XPATH_GENERAL_COMMENT_SECTIONS, XPATH_REPLY_COMMENT_SECTIONS
    )

    # 返信の数を取得
    reply_count = int(
        webelement.find_element(
            By.XPATH, RELATIVE_XPATH_GENERAL_COMMENT_REPLY_COUNT
        ).text
    )

    # 返信表示ボタンをクリック
    webelement.find_element(
        By.XPATH, RELATIVE_XPATH_GENERAL_COMMENT_REPLY_BUTTON
    ).click()

    reply_comments = []
    cnt = 10
    while (
        webelement.find_elements(
            By.XPATH, RELATIVE_XPATH_GENERAL_COMMENT_REPLY_BUTTON_MORE
        )
        or cnt < max_replies
    ):
        # もっと返信を表示するボタンをクリック
        webelement.find_element(
            By.XPATH, RELATIVE_XPATH_GENERAL_COMMENT_REPLY_BUTTON_MORE
        ).click()
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
) -> dict[str, str | list[GeneralComment]]:
    """
    記事のコメントをスクレイピングする関数。

    Parameters
    ----------
    url : str
        記事のコメントページのURL
    max_comments : int
        取得するコメントの最大数。10の倍数以外の数を入力すると、10の倍数に切り上げて処理される。
    max_replies : int
        取得する返信の最大数
    order : str, Optional
        コメントの表示順。 "newer" または "recommended" のいずれかを指定
    timeout : int, Optional
        WebDriverのタイムアウト時間

    Returns
    -------
    dict[str, str | list[GeneralComment]]
        スクレイピングしたデータ
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
