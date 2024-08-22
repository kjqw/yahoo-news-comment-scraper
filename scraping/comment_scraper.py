import json
import re
from pathlib import Path, PurePosixPath

import selenium
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


def init_driver(TIMEOUT: int = 10) -> webdriver:
    """
    WebDriverを初期化する

    Returns
    -------
    driver : webdriver
        WebDriverオブジェクト
    """
    # Chromeオプションの設定
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    )

    # WebDriverの設定
    driver = webdriver.Remote(
        command_executor="http://selenium:4444/wd/hub", options=chrome_options
    )
    driver.set_page_load_timeout(TIMEOUT)

    return driver


def get_comment_sections(
    driver: webdriver,
) -> list[selenium.webdriver.remote.webelement.WebElement]:
    """
    コメントセクションのリストを取得する

    Parameters
    ----------
    driver : webdriver
        記事のコメントページを開いているWebDriverオブジェクト

    Returns
    -------
    articles : list[selenium.webdriver.remote.webelement.WebElement]
        コメントセクションのリスト
    """

    # class="viewableWrapper"を持つarticleタグを取得
    articles = driver.find_elements(By.CSS_SELECTOR, "article.viewableWrapper")

    return articles


def read_xpath_json(file_path: Path) -> dict:
    """
    指定されたパスからXPathが記述されたJSONファイルを読み込み、辞書形式で返す

    Parameters
    ----------
    file_path : Path
        読み込むJSONファイルのパス

    Returns
    -------
    dict
        JSONファイルを読み込んだ辞書
    """
    with file_path.open("r", encoding="utf-8") as f:
        xpath_dict = json.load(f)

    return xpath_dict


def find_all_combinations(
    driver: webdriver, base_xpath: str
) -> tuple[list[str], list[selenium.webdriver.remote.webelement.WebElement]]:
    """
    指定されたXPathで[i\d+]のすべての組み合わせに一致する要素と、そのXPathを取得する

    Parameters
    ----------
    driver : webdriver
        記事のコメントページを開いているWebDriverオブジェクト
    base_xpath : str
        [i\d+]のプレースホルダを含むXPathのベースパス

    Returns
    -------
    tuple[list[str], list[selenium.webdriver.remote.webelement.WebElement]]
        該当するすべてのXPathと、その要素のリスト
    """
    elements_with_xpath = []

    # [i\d+]に一致する部分を全て見つける
    placeholders = re.findall(r"\[i\d+\]", base_xpath)

    def recursive_find(current_xpath: str, index: int) -> None:
        """
        再帰的にプレースホルダを置換して要素を探索する

        Parameters
        ----------
        current_xpath : str
            現在のXPath
        index : int
            現在のプレースホルダのインデックス
        """
        # ベースケース: すべてのプレースホルダを置換した場合
        if index >= len(placeholders):
            found_elements = driver.find_elements(By.XPATH, current_xpath)
            if found_elements:
                elements_with_xpath.extend(
                    (current_xpath, element) for element in found_elements
                )
            return

        # 置換するプレースホルダを取り出す
        placeholder = placeholders[index]
        i = 1

        while True:
            # プレースホルダを実際のインデックスで置換
            xpath_with_index = current_xpath.replace(placeholder, f"[{i}]", 1)
            found_elements = driver.find_elements(By.XPATH, xpath_with_index)

            if not found_elements:
                break

            # 次のプレースホルダの組み合わせを再帰的に探索
            recursive_find(xpath_with_index, index + 1)
            i += 1

    # 最初のプレースホルダから探索を開始
    recursive_find(base_xpath, 0)

    # XPathと要素をそれぞれのリストに分解
    xpaths, elements = zip(*elements_with_xpath) if elements_with_xpath else ([], [])

    return list(xpaths), list(elements)


def push_reply_display_button(driver: webdriver, button_xpath: str) -> None:
    """
    返信表示ボタンを押す

    Parameters
    ----------
    driver : webdriver
        記事のコメントページを開いているWebDriverオブジェクト
    button_xpath : str
        返信表示ボタンのXPath
    """

    xpaths, elements = find_all_combinations(driver, button_xpath)
    for xpath in xpaths:
        try:
            element = driver.find_element(By.XPATH, xpath)
            element.click()
            print(f"Clicked: {xpath}")
        except NoSuchElementException:
            continue
        except TimeoutException:
            continue


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


if __name__ == "__main__":
    # 使用例
    base_xpath = "/html/body/div[1]/div[2]"
    target_xpath = "/html/body/div[1]/div[2]/span"

    print(get_relative_xpath(base_xpath, target_xpath))
