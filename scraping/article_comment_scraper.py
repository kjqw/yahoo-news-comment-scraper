import pickle
from collections import defaultdict
from pathlib import Path

import utils
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from xpath_article_page import *


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
    relative_xpath_reply_comment_sections = utils.get_relative_xpath(
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

        print("もっと返信を表示して待機中")

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

        print(f"返信コメントを追加で取得した。現在の返信数: {cnt}")

    # 返信コメントのセクションを取得
    reply_comment_sections = webelement.find_elements(
        By.XPATH, relative_xpath_reply_comment_sections
    )
    print(f"返信コメントの数: {len(reply_comment_sections)}")
    print()
    return reply_comment_sections


def get_article_comments(
    url: str,
    max_comments: int,
    max_replies: int,
    order: str = "recommended",
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
        取得する返信の最大数。10の倍数以外の数を入力すると、10の倍数に切り上げて処理される。
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

        # コメント数を取得
        total_comment_count_with_reply, total_comment_count_without_reply = (
            get_total_comment_count(driver)
        )
        data["total_comment_count_with_reply"] = total_comment_count_with_reply
        data["total_comment_count_without_reply"] = total_comment_count_without_reply

        # 最大数に達するまでコメントを取得
        page = 1
        data["comments"] = []
        while (page - 1) * 10 < max_comments:
            # 一般コメントのセクションを取得
            # general_comment_sections = utils.get_comment_sections(driver)
            general_comment_sections = driver.find_elements(
                By.XPATH, XPATH_GENERAL_COMMENT_SECTIONS
            )
            # それぞれの一般コメントについて処理
            for general_comment_section in general_comment_sections:

                # 一般コメントの情報を取得
                tmp_dict = {}
                for key, xpath in xpaths_general_comments.items():
                    tmp_dict[key] = general_comment_section.find_element(
                        By.XPATH, xpath
                    ).text

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

                # 返信コメントのセクションを取得
                reply_comment_sections = get_reply_comment_sections(
                    general_comment_section, max_replies
                )

                # 返信コメントがない場合はスキップ
                if reply_comment_sections:
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
    url = "https://news.yahoo.co.jp/articles/a9e7e7f9c3f25c2becdefa309c22e1f8cb60240f/comments"
    max_comments = 20
    max_replies = 20

    # コメントを取得
    data = get_article_comments(url, max_comments, max_replies)

    # データを保存
    save_path = Path(__file__).parent / "data" / "comments.pkl"
    save_data(data, save_path)
