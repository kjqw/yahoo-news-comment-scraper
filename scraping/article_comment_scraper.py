import argparse
import pickle
from collections import defaultdict
from math import ceil
from pathlib import Path

import functions
from classes import GeneralComment, ReplyComment
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
from xpaths.xpath_article_comment_page import *


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
            "user_link": RELATIVE_XPATH_GENERAL_COMMENT_USERNAME,
            "posted_time": RELATIVE_XPATH_GENERAL_COMMENT_POSTED_TIME,
            "comment_text": RELATIVE_XPATH_GENERAL_COMMENT_COMMENT_TEXT,
            "agreements": RELATIVE_XPATH_GENERAL_COMMENT_AGREEMENTS,
            "acknowledgements": RELATIVE_XPATH_GENERAL_COMMENT_ACKNOWLEDGEMENTS,
            "disagreements": RELATIVE_XPATH_GENERAL_COMMENT_DISAGREEMENTS,
            "reply_count": RELATIVE_XPATH_GENERAL_COMMENT_REPLY_COUNT,
        }
        xpaths_reply_comments = {
            "username": RELATIVE_XPATH_REPLY_COMMENT_USERNAME,
            "user_link": RELATIVE_XPATH_REPLY_COMMENT_USERNAME,
            "posted_time": RELATIVE_XPATH_REPLY_COMMENT_POSTED_TIME,
            "comment_text": RELATIVE_XPATH_REPLY_COMMENT_COMMENT_TEXT,
            "agreements": RELATIVE_XPATH_REPLY_COMMENT_AGREEMENTS,
            "acknowledgements": RELATIVE_XPATH_REPLY_COMMENT_ACKNOWLEDGEMENTS,
            "disagreements": RELATIVE_XPATH_REPLY_COMMENT_DISAGREEMENTS,
        }

        # ドライバを初期化
        driver = functions.init_driver(timeout)

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
            general_comment_sections = driver.find_elements(
                By.XPATH, XPATH_GENERAL_COMMENT_SECTIONS
            )
            # それぞれの一般コメントについて処理
            for general_comment_section in tqdm(
                general_comment_sections,
                desc=f"コメント取得 {page}/{ceil(max_comments//10)}ページ目",
            ):
                # 一般コメントのオブジェクトを作成
                general_comment = GeneralComment()

                # 一般コメントの情報を取得
                general_comment.get_info(
                    general_comment_section, xpaths_general_comments
                )

                # 返信コメントのセクションを取得
                reply_comment_sections = get_reply_comment_sections(
                    general_comment_section, max_replies
                )

                # 返信コメントがない場合はスキップ
                if reply_comment_sections:
                    # 返信コメントの情報を取得
                    for reply_comment_section in reply_comment_sections:
                        # 返信コメントのオブジェクトを作成
                        reply_comment = ReplyComment()

                        # 返信コメントの情報を取得
                        reply_comment.get_info(
                            reply_comment_section, xpaths_reply_comments
                        )
                        reply_comment.base_comment = general_comment

                        # 一般コメントに返信コメントを追加
                        general_comment.reply_comments.append(reply_comment)

                # 一般コメントを格納
                data["comments"].append(general_comment)

            # 次のページに移動
            page += 1
            functions.open_page(driver, url, page, order)

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


def main(
    url: str,
    max_comments: int,
    max_replies: int,
    order: str,
    timeout: int,
    save_path: Path,
) -> None:
    # コメントを取得
    data = get_article_comments(url, max_comments, max_replies, order, timeout)

    # データを保存
    save_data(data, save_path)


if __name__ == "__main__":
    # デフォルトの値
    default_url = "https://news.yahoo.co.jp/articles/a9e7e7f9c3f25c2becdefa309c22e1f8cb60240f/comments"
    default_url = "https://news.yahoo.co.jp/articles/0db50721a0f1d89d7e42a3d74d12e7bbc89d3ce8/comments"
    default_max_comments = 20
    default_max_replies = 20
    default_order = "recommended"
    default_timeout = 10
    default_save_path = Path(__file__).parent / "data" / "comments.pkl"

    parser = argparse.ArgumentParser(
        description="記事のコメントを取得し、保存するスクリプト"
    )

    # 引数を追加（デフォルト値を指定）
    parser.add_argument(
        "--url", default=default_url, help="コメントを取得する記事のURL"
    )
    parser.add_argument(
        "--max_comments",
        type=int,
        default=default_max_comments,
        help="取得するコメントの最大数",
    )
    parser.add_argument(
        "--max_replies",
        type=int,
        default=default_max_replies,
        help="取得するリプライの最大数",
    )
    parser.add_argument(
        "--order",
        type=str,
        default=default_order,
        help="コメントの表示順",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=default_timeout,
        help="WebDriverのタイムアウト時間",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=default_save_path,
        help="保存するファイルのパス",
    )

    # 引数を解析
    args = parser.parse_args()

    # メイン処理を実行
    main(
        args.url,
        args.max_comments,
        args.max_replies,
        args.order,
        args.timeout,
        args.save_path,
    )
