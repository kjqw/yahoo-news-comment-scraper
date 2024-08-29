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
        # スクレイピングの結果を格納する辞書を初期化
        data = defaultdict(str)

        # XPathの辞書を作成
        xpaths_article = {
            "article_title": XPATH_ARTICLE_TITLE,
            "article_author": XPATH_ARTICLE_AUTHOR,
            "article_posted_time": XPATH_ARTICLE_POSTED_TIME,
            "total_comment_count_with_reply": XPATH_TOTAL_COMMENT_COUNT_WITH_REPLY,
            "total_comment_count_without_reply": XPATH_TOTAL_COMMENT_COUNT_WITHOUT_REPLY,
        }
        xpaths_general_comments = {
            "username": RELATIVE_XPATH_GENERAL_COMMENT_USERNAME,
            "posted_time": RELATIVE_XPATH_GENERAL_COMMENT_POSTED_TIME,
            "comment_text": RELATIVE_XPATH_GENERAL_COMMENT_COMMENT_TEXT,
            "agreements": RELATIVE_XPATH_GENERAL_COMMENT_AGREEMENTS,
            "acknowledgements": RELATIVE_XPATH_GENERAL_COMMENT_ACKNOWLEDGEMENTS,
            "disagreements": RELATIVE_XPATH_GENERAL_COMMENT_DISAGREEMENTS,
            "reply_count": RELATIVE_XPATH_GENERAL_COMMENT_REPLY_COUNT,
        }
        xpaths_reply_comments = {
            "username": RELATIVE_XPATH_REPLY_COMMENT_USERNAME,
            "posted_time": RELATIVE_XPATH_REPLY_COMMENT_POSTED_TIME,
            "comment_text": RELATIVE_XPATH_REPLY_COMMENT_COMMENT_TEXT,
            "agreements": RELATIVE_XPATH_REPLY_COMMENT_AGREEMENTS,
            "acknowledgements": RELATIVE_XPATH_REPLY_COMMENT_ACKNOWLEDGEMENTS,
            "disagreements": RELATIVE_XPATH_REPLY_COMMENT_DISAGREEMENTS,
        }

        # ドライバを初期化
        driver = utils.init_driver(timeout)

        # 記事のコメントページを開く
        driver.get(url)

        # 記事の情報を取得
        for key, xpath in xpaths_article.items():
            data[key] = driver.find_element(By.XPATH, xpath).text

        # 最大数に達するまでコメントを取得
        page = 1
        data["comments"] = []
        while (page - 1) * 10 < max_comments:
            # 一般コメントのセクションを取得
            general_comment_sections = utils.get_comment_sections(driver)
            # それぞれの一般コメントについて処理
            for comment_section in general_comment_sections:
                # 返信コメントのセクションを取得
                reply_comment_sections = get_reply_comment_sections(
                    comment_section, max_replies
                )

                # 一般コメントの情報を取得
                tmp_dict = {}
                for key, xpath in xpaths_general_comments.items():
                    tmp_dict[key] = comment_section.find_element(By.XPATH, xpath).text

                # 一般コメントのオブジェクトを作成
                general_comment = GeneralComment(
                    username=tmp_dict["username"],
                    posted_time=tmp_dict["posted_time"],
                    comment_text=tmp_dict["comment_text"],
                    agreements=tmp_dict["agreements"],
                    acknowledgements=tmp_dict["acknowledgements"],
                    disagreements=tmp_dict["disagreements"],
                    reply_count=tmp_dict["reply_count"],
                )

                # 返信コメントの情報を取得
                for reply_comment_section in reply_comment_sections:
                    tmp_dict = {}
                    for key, xpath in xpaths_reply_comments.items():
                        tmp_dict[key] = reply_comment_section.find_element(
                            By.XPATH, xpath
                        ).text

                    # 返信コメントのオブジェクトを作成
                    reply_comment = ReplyComment(
                        username=tmp_dict["username"],
                        posted_time=tmp_dict["posted_time"],
                        comment_text=tmp_dict["comment_text"],
                        agreements=tmp_dict["agreements"],
                        acknowledgements=tmp_dict["acknowledgements"],
                        disagreements=tmp_dict["disagreements"],
                    )
                    general_comment.reply_comments.append(reply_comment)
                    reply_comment.set_base_comment(general_comment)

                # 一般コメントを格納
                data["comments"].append(general_comment)

            # 次のページに移動
            page += 1
            utils.open_page(driver, url, page, order)

        return data

    finally:
        driver.quit()


def save_data(data: dict[str, str | list[GeneralComment]], save_path: Path) -> None:
    """
    スクレイピングしたデータを保存する関数。

    Parameters
    ----------
    data : dict[str, str | list[GeneralComment]]
        スクレイピングしたデータ
    save_path : Path
        保存先のパス
    """
    # 保存先のディレクトリが存在しない場合は作成
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    url = "https://news.yahoo.co.jp/articles/ddae7ca185c389a92d2c1136a699a32fe4415094/comments"
    max_comments = 10
    max_replies = 5

    comments = get_article_comments(url, max_comments, max_replies)
    print(comments)
