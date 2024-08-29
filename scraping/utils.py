import inspect
import re
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import xpath
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement


def init_driver(TIMEOUT: int = 10) -> webdriver:
    """
    WebDriverを初期化する

    Parameters
    ----------
    TIMEOUT : int, optional
        ページのロード待ち時間（秒）を設定, デフォルトは10秒

    Returns
    -------
    driver : webdriver
        WebDriverオブジェクト
    """
    # Chromeオプションの設定
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # ヘッドレスモードを有効にする（GUIなし）
    chrome_options.add_argument("--no-sandbox")  # サンドボックスモードを無効にする
    chrome_options.add_argument(
        "--disable-dev-shm-usage"
    )  # 共有メモリの使用を無効にする
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    )  # ユーザーエージェントを設定

    # WebDriverの設定
    driver = webdriver.Remote(
        command_executor="http://selenium:4444/wd/hub", options=chrome_options
    )
    driver.set_page_load_timeout(TIMEOUT)  # ページロードのタイムアウトを設定

    return driver


def get_comment_sections(driver: webdriver) -> list[WebElement]:
    """
    コメントセクションの親要素のリストを取得する

    Parameters
    ----------
    driver : webdriver
        記事のコメントページを開いているWebDriverオブジェクト

    Returns
    -------
    parent_sections : list[WebElement]
        コメントセクションの親要素のリスト
    """
    # class="viewableWrapper"を持つarticleタグを取得
    articles = driver.find_elements(By.CSS_SELECTOR, "article.viewableWrapper")

    # 各コメントセクションの親要素を取得
    parent_sections = [article.find_element(By.XPATH, "..") for article in articles]

    return parent_sections


def list_to_xpath(base_xpath: str, current_indices: list[int]) -> str:
    """
    与えられたリストを元に、base_xpathの[i\d+]を置換してXPathを生成する関数。

    Parameters
    ----------
    base_xpath : str
        プレースホルダを含むベースのXPath
    current_indices : list[int]
        プレースホルダを置換するためのインデックスリスト

    Returns
    -------
    str
        インデックスで置換されたXPath
    """
    for i, current_index in enumerate(current_indices):
        base_xpath = base_xpath.replace(
            f"[i{i+1}]", f"[{current_index}]"
        )  # プレースホルダをインデックスで置換
    return base_xpath


def generate_next_list(
    current_indices: list[int], max_indices: list[int]
) -> list[int] | None:
    """
    与えられたリストを次の組み合わせに更新する関数。

    Parameters
    ----------
    current_indices : list[int]
        現在のインデックスのリスト
    max_indices : list[int]
        各プレースホルダの最大値リスト

    Returns
    -------
    list[int] | None
        次のインデックスリスト、最大値を超えた場合はNoneを返す
    """
    # 現在のインデックスの一部が最大値を超えているか、全てが最大値に達している場合はNoneを返す
    if (
        any(
            current_index > max_current_index
            for current_index, max_current_index in zip(current_indices, max_indices)
        )
        or current_indices == max_indices
    ):
        return None

    # 末尾から始めて桁上げを行う
    for i in range(len(current_indices) - 1, -1, -1):
        if current_indices[i] < max_indices[i]:
            current_indices[i] += 1  # インデックスをインクリメント
            break
        else:
            current_indices[i] = 1  # インデックスをリセット

    return current_indices


def find_all_combinations(
    driver: webdriver, base_xpath: str, max_indices: list[int]
) -> tuple[list[str], list[WebElement]]:
    """
    指定されたXPathで[i\d+]のすべての組み合わせに一致する要素と、そのXPathを取得する。

    Parameters
    ----------
    driver : webdriver
        記事のコメントページを開いているWebDriverオブジェクト
    base_xpath : str
        [i\d+]のプレースホルダを含むXPathのベースパス
    max_indices : list[int]
        プレースホルダの最大値のリスト

    Returns
    -------
    tuple[list[str], list[WebElement]]
        該当するすべてのXPathと、その要素のリスト。
    """

    def generate_xpath_combinations(
        base_xpath: str, max_indices: list[int]
    ) -> list[str]:
        """
        すべてのインデックスの組み合わせに基づいてXPathを生成する。

        Parameters
        ----------
        base_xpath : str
            [i\d+]のプレースホルダを含むXPathのベースパス。
        max_indices : list[int]
            プレースホルダの最大値のリスト。

        Returns
        -------
        list[str]
            生成されたすべてのXPathのリスト。
        """
        # プレースホルダの部分を正規表現で取得
        placeholders = re.findall(r"\[i(\d+)\]", base_xpath)

        # 数字部分だけを取得して、整数に変換
        placeholder_indices = [int(match) for match in placeholders]
        max_placeholder_index = max(placeholder_indices)

        # プレースホルダの数だけ1で初期化したリストを生成
        current_indices = [1 for _ in range(max_placeholder_index)]
        xpaths = []

        # すべての組み合わせを生成
        while current_indices is not None:
            xpath = list_to_xpath(base_xpath, current_indices)
            xpaths.append(xpath)  # 生成されたXPathをリストに追加
            current_indices = generate_next_list(current_indices, max_indices)

        # 重複を削除してソート
        xpaths = sorted(list(set(xpaths)))

        return xpaths

    # すべてのXPathを生成
    all_xpaths = generate_xpath_combinations(base_xpath, max_indices)

    # 生成されたXPathに一致する要素を検索
    found_elements = []
    matching_xpaths = []

    for xpath in all_xpaths:
        elements = driver.find_elements(By.XPATH, xpath)  # 各XPathに一致する要素を検索
        if elements:
            matching_xpaths.append(xpath)
            found_elements.append(elements)

    return matching_xpaths, found_elements


def get_relative_xpath(base_xpath: str, target_xpath: str) -> str:
    """
    base_xpathから見たtarget_xpathの相対パスを返す関数

    Parameters
    ----------
    base_xpath : str
        基準となるXPath
    target_xpath : str
        相対パスを計算したいXPath

    Returns
    -------
    str
        base_xpathから見たtarget_xpathの相対パス

    Raises
    ------
    ValueError
        base_xpathまたはtarget_xpathが不正な場合
    """
    # base_xpathとtarget_xpathをパースしてリストに分割
    base_path = PurePosixPath(base_xpath)
    target_path = PurePosixPath(target_xpath)

    # 共通のパス部分を取り除いて相対パスを計算
    relative_path = target_path.relative_to(base_path)

    # PurePosixPathオブジェクトをXPath形式の文字列に変換
    return str(relative_path)


def open_page(driver: webdriver, url: str, order: str, page: int) -> None:
    """
    ページを開く関数

    Parameters
    ----------
    driver : webdriver
        WebDriverオブジェクト
    url : str
        開くページのURL
    order : str
        ソート順
    page : int
        開くページ番号
    """
    # URLを解析
    url_parts = urlparse(url)

    # 既存のクエリパラメータを解析
    query_params = parse_qs(url_parts.query)

    # 新しいパラメータを追加または更新
    query_params["order"] = [order]
    query_params["page"] = [str(page)]

    # 新しいクエリ文字列を生成
    new_query = urlencode(query_params, doseq=True)

    # 新しいURLを構築
    new_url_parts = url_parts._replace(query=new_query)
    new_url = urlunparse(new_url_parts)

    # ページを開く
    driver.get(new_url)
