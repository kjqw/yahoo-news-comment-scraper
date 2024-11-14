import inspect
import re
from pathlib import PurePosixPath
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import psycopg2
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from xpaths import xpath_article_comment_page


def init_driver(TIMEOUT: int = 10) -> webdriver:
    """
    WebDriverを初期化する。

    Parameters
    ----------
    TIMEOUT : int, Optional
        ページのロード待ち時間（秒）を設定する

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
    コメントセクションの親要素のリストを取得する。

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
    r"""
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
        次のインデックスリスト。最大値を超えた場合はNoneを返す。
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
    r"""
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
        該当するすべてのXPathと、その要素のリスト
    """

    def generate_xpath_combinations(
        base_xpath: str, max_indices: list[int]
    ) -> list[str]:
        r"""
        すべてのインデックスの組み合わせに基づいてXPathを生成する。

        Parameters
        ----------
        base_xpath : str
            [i\d+]のプレースホルダを含むXPathのベースパス
        max_indices : list[int]
            プレースホルダの最大値のリスト

        Returns
        -------
        list[str]
            生成されたすべてのXPathのリスト
        """
        # プレースホルダの部分を正規表現で取得
        placeholders = re.findall(r"\[i(\d+)\]", base_xpath)

        # プレースホルダが存在しない場合は、base_xpathをそのままリストにして返す
        if not placeholders:
            return [base_xpath]

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
            found_elements.extend(elements)

    return matching_xpaths, found_elements


def get_relative_xpath(base_xpath: str, target_xpath: str) -> str:
    """
    base_xpathから見たtarget_xpathの相対パスを返す関数。

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
    """
    # base_xpathとtarget_xpathをパースしてリストに分割
    base_path = PurePosixPath(base_xpath)
    target_path = PurePosixPath(target_xpath)

    # 共通のパス部分を取り除いて相対パスを計算
    relative_path = target_path.relative_to(base_path)

    # PurePosixPathオブジェクトをXPath形式の文字列に変換
    return str(relative_path)


def open_page(driver: webdriver, url: str, page: int, order: str = "") -> None:
    """
    指定されたURLのページを開く。

    Parameters
    ----------
    driver : WebDriver
        WebDriverオブジェクト
    url : str
        開くページのURL
    page : int
        開くページ番号。正の整数でない場合はページ番号を変更しない。
    order : str, Optional
        コメントの表示順。'newer' または 'recommended' の場合にのみ適用される。

    Notes
    -----
    無効な `order` 値が指定された場合、'order' パラメータは変更されず、
    無効な `page` 値が指定された場合、'page' パラメータは変更されません。
    無効な値が入力された場合には、標準出力にその旨が表示されます。
    """
    # 有効なorder値のセット
    valid_orders = {"newer", "recommended"}

    # orderが有効な値か確認
    if order and order not in valid_orders:
        print(f"無効な`order`値: {order}")
        print("有効な値は次のいずれかです: ", valid_orders)
        order = ""

    # pageが正の整数か確認
    if not isinstance(page, int) or page <= 0:
        print(f"無効な`page`値: {page}")
        print("`page`は正の整数である必要があります。")
        page = 0

    # URLを解析してクエリパラメータを取得
    url_parts = urlparse(url)
    query_params = parse_qs(url_parts.query)

    # orderが有効な場合にクエリパラメータを更新
    if order:
        query_params["order"] = [order]

    # pageが有効な場合にクエリパラメータを更新
    if page:
        query_params["page"] = [str(page)]

    # 新しいクエリ文字列を生成してURLを更新
    new_query = urlencode(query_params, doseq=True)
    new_url_parts = url_parts._replace(query=new_query)
    new_url = urlunparse(new_url_parts)

    # 新しいURLでページを開く
    driver.get(new_url)


def get_filtered_vars(pattern: str = None) -> dict[str, str]:
    """
    xpath.pyから変数を取得し、指定の正規表現パターンに基づいてフィルタリングする。

    Parameters
    ----------
    pattern : str
        変数名をフィルタリングするための正規表現パターン。Noneの場合、全ての変数を含む。

    Returns
    -------
    dict[str, str]
        フィルタリングされた変数の名前と値の辞書
    """
    # モジュール内の変数を取得
    module_vars = [
        (name, value)
        for name, value in inspect.getmembers(xpath_article_comment_page)
        if not name.startswith("_")
    ]

    # パターンが指定されていない場合は全ての変数を返す
    if pattern is None:
        return dict(module_vars)

    # フィルタリング
    filtered_vars_values = {
        name: value for name, value in module_vars if re.search(pattern, name)
    }

    return filtered_vars_values


def normalize_number(text: str) -> int:
    """
    数字表記を正規化し、単位（件、個など）を削除する関数。

    Parameters
    ----------
    text : str
        正規化対象のテキスト

    Returns
    -------
    int
        正規化後の数値
    """

    # 1.3万、2.5千などの表記を変換
    text = re.sub(
        r"(\d+(\.\d+)?)万", lambda x: str(int(float(x.group(1)) * 10000)), text
    )
    text = re.sub(
        r"(\d+(\.\d+)?)千", lambda x: str(int(float(x.group(1)) * 1000)), text
    )

    # 1,234のようなカンマ区切りを削除
    text = re.sub(r"(\d+),(\d+)", r"\1\2", text)

    # 単位（件、個など）を削除
    text = re.sub(r"(\d+)(件|個|円|人|回|分|時間)?", r"\1", text)

    return int(text)
