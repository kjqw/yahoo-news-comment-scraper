import functions
from classes import DBBase
from selenium.webdriver.common.by import By
from tqdm import tqdm
from xpaths.xpath_ranking_page import *


def get_articles(url: str = URL_COMMENT_RANKING, timeout: int = 10) -> None:
    """
    記事のリンクを取得する。

    Parameters
    ----------
    url : str, Optional
        記事のリンクを取得するページのURL。デフォルトではコメント数ランキングのページ。
    timeout : int, Optional
        WebDriverのタイムアウト時間
    """
    try:
        # ドライバを初期化
        driver = functions.init_driver(timeout)

        # ページを開く
        driver.get(url)

        # 記事の要素を取得
        for element in tqdm(driver.find_elements(By.XPATH, XPATH_ARTICLE_LINKS)):
            article = DBBase()
            article.get_info(
                element,
                {
                    "article_link": ".",
                    "article_title": RELATIVE_XPATH_ARTICLE_TITLE,
                    "author": RELATIVE_XPATH_ARTICLE_AUTHOR,
                    "posted_time": RELATIVE_XPATH_ARTICLE_POSTED_TIME,
                    "ranking": RELATIVE_XPATH_RANKING,
                    "comment_count_per_hour": RELATIVE_XPATH_ARTICLE_COMMENT_COUNT_PER_HOUR,
                },
            )

            # 数値を正規化
            article.normalize_number()

            # データベースに保存
            article.save_data("articles")

    finally:
        driver.quit()


if __name__ == "__main__":
    get_articles()
