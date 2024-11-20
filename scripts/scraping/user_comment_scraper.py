# %%
import functions
from classes import DBBase
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
from xpaths.xpath_user_page import *


def get_articles_and_comments(
    url: str,
    max_comments: int,
    timeout: int = 10,
):
    """
    ユーザーのページから、ユーザーがどの記事にどのようなコメントをしているかを取得する。

    Parameters
    ----------
    url : str
        ユーザーのページのURL
    timeout : int, Optional
        WebDriverのタイムアウト時間
    """

    try:
        # ドライバを初期化
        driver = functions.init_driver(timeout)

        # ページを開く
        driver.get(url)

        user = DBBase()

        # 基本情報を取得
        user.user_link = url
        user.get_info(
            driver,
            {
                "username": XPATH_USER_NAME,
                "total_comment_count": XPATH_TOTAL_COMMENT_COUNT,
                "total_agreements": XPATH_TOTAL_AGREEMENTS,
                "total_acknowledgements": XPATH_TOTAL_ACKNOWLEDGEMENTS,
                "total_disagreements": XPATH_TOTAL_DISAGREEMENTS,
            },
        )

        # 「もっと見る」ボタンが表示されるまで待機
        WebDriverWait(driver, timeout).until(
            EC.presence_of_all_elements_located((By.XPATH, XPATH_COMMENT_MORE_BUTTON))
        )
        cnt = 0
        while cnt < max_comments // 10 and driver.find_elements(
            By.XPATH, XPATH_COMMENT_MORE_BUTTON
        ):

            # 「もっと見る」ボタンをクリック
            driver.find_element(By.XPATH, XPATH_COMMENT_MORE_BUTTON).click()

            # 次の「もっと見る」ボタンが表示されるまで待機
            WebDriverWait(driver, timeout).until(
                EC.presence_of_all_elements_located(
                    (By.XPATH, XPATH_COMMENT_MORE_BUTTON)
                )
            )

            cnt += 1
            print(f"{cnt}回目の「もっと見る」ボタンをクリックしました")

        # コメントセクションを取得
        WebDriverWait(driver, timeout).until(
            EC.presence_of_all_elements_located((By.XPATH, XPATH_COMMENT_SECTIONS))
        )

        comment_sections = driver.find_elements(By.XPATH, XPATH_COMMENT_SECTIONS)
        return comment_sections

    finally:
        driver.quit()


# %%
if __name__ == "__main__":
    url = "https://news.yahoo.co.jp/users/9BDLMjtCapOdtyDb4scwLvKCSCm4G-CwFHqKB2o97ej0XkV200"
    max_comments = 30

    result = get_articles_and_comments(url, max_comments)
    print(result)
# %%
len(result)
# %%
