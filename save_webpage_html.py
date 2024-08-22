import re
from pathlib import Path

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def sanitize_filename(filename: str) -> str:
    """
    ファイル名に使用できない文字を除去する関数。

    Parameters
    ----------
    filename : str
        元のファイル名。

    Returns
    -------
    str
        使用可能なファイル名。
    """
    return re.sub(r'[\\/*?:"<>|]', "", filename)


def save_html(url: str, save_path: Path, timeout: int = 30) -> None:
    """
    指定したURLのHTMLを取得してローカルファイルに保存する関数。

    Parameters
    ----------
    url : str
        取得するWebページのURL。
    save_path : Path
        保存するディレクトリ。
    timeout : int
        ページロードのタイムアウト時間（秒）。
    """
    try:
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
        driver.set_page_load_timeout(timeout)

        # URLにアクセスして完全にロードされるまで待機
        driver.get(url)

        # ページソースを取得
        html = driver.page_source
        driver.quit()

        # HTMLをパース
        soup = BeautifulSoup(html, "html.parser")

        # ページタイトルを取得してファイル名を生成
        title = soup.title.string if soup.title else "default_title"
        sanitized_title = sanitize_filename(title)
        file_path = save_path / f"{sanitized_title}.html"

        # パースしたHTMLをローカルファイルに保存
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(soup.prettify())

        print(f"HTMLを{file_path}に保存しました。")

    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    # 使用例
    # url = "https://news.yahoo.co.jp/articles/f597a958b084e6e9501b579003e3f48460ffcb7c"
    url = "https://news.yahoo.co.jp/users/WxsosrxMy0hRXtWoFVIj_CfD22qV_QuKArlWzVXjvIdbVab200"
    save_path = Path(__file__).parent / "html"  # 保存するディレクトリ
    Path(save_path).mkdir(exist_ok=True)  # ディレクトリが存在しない場合は作成する
    save_html(url, save_path)
