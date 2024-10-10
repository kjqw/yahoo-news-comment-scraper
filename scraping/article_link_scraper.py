import functions
from classes import Article
from selenium.webdriver.common.by import By
from xpaths.xpath_ranking_page import *


def get_articles(url: str = URL_COMMENT_RANKING, timeout: int = 10) -> list[Article]:
    """
    記事のリンクを取得する。

    Parameters
    ----------
    url : str, Optional
        記事のリンクを取得するページのURL。デフォルトではコメント数ランキングのページ。
    timeout : int, Optional
        WebDriverのタイムアウト時間

    Returns
    -------
    list[Article]
        記事のクラスのリスト
    """
    try:
        # ドライバを初期化
        driver = functions.init_driver(timeout)

        # ページを開く
        driver.get(url)

        # 記事の要素を取得
        articles = []
        for element in driver.find_elements(By.CLASS_NAME, CLASS_ARTICLE_LINKS):
            article = Article()
            article.get_info(
                element,
                {
                    "article_link": ".",
                    "article_title": RELATIVE_XPATH_ARTICLE_TITLE,
                    "author": RELATIVE_XPATH_ARTICLE_AUTHOR,
                    "posted_time": RELATIVE_XPATH_ARTICLE_POSTED_TIME,
                    "ranking": RELATIVE_XPATH_RANKING,
                },
            )

            articles.append(article)

        return articles

    finally:
        driver.quit()
