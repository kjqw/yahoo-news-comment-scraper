import sys
from datetime import datetime
from pathlib import Path

import functions
from selenium import webdriver
from selenium.webdriver.common.by import By

sys.path.append(str(Path(__file__).parents[1]))

import db_manager


class DBBase:
    """
    データベースに保存できる基底クラス。
    """

    def export_data(self) -> dict[str, str | int | datetime | None]:
        """
        インスタンスのすべての属性を辞書形式で返す。
        """
        return {k: v for k, v in self.__dict__.items()}

    def save_data(
        self,
        table_name: str,
        db_config: dict = db_manager.DB_CONFIG,
    ) -> None:
        """
        データベースに保存する。

        Parameters
        ----------
        table_name : str
            データを保存するテーブルの名前。
        db_config : dict[str, str]
            データベース接続に必要な設定情報。
        """
        data = self.export_data()
        columns = ", ".join(data.keys())
        values = ", ".join([f"'{value}'" for value in data.values()])

        query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"

        db_manager.execute_query(query, db_config, commit=True)

    def update_data(
        self,
        table_name: str,
        db_config: dict = db_manager.DB_CONFIG,
    ) -> None:
        """
        データベースのデータを更新する。

        Parameters
        ----------
        table_name : str
            データを保存するテーブルの名前。
        db_config : dict[str, str]
            データベース接続に必要な設定情報。
        """
        data = self.export_data()
        columns = ", ".join(data.keys())
        values = ", ".join([f"'{value}'" for value in data.values()])

        query = f"UPDATE {table_name} SET ({columns}) = ({values}) WHERE article_link = '{self.article_link}'"

        db_manager.execute_query(query, db_config, commit=True)

    def get_info(self, driver: webdriver, xpaths: dict[str, list[str] | str]) -> None:
        """
        渡されたXPATHの辞書を用いて記事の情報を取得する。
        '_link'で終わるキーの場合はhref属性を、それ以外の場合はtextを取得する。

        Parameters
        ----------
        driver : webdriver
            WebDriverインスタンス
        xpaths : dict[str, list[str] | str]
            XPATHの辞書（キーに対して単一または複数のXPATHを渡す）
        """
        try:
            # 各XPATHを用いて情報を取得
            for key, xpath_value in xpaths.items():
                element = None

                # 値が文字列の場合はリストに変換
                xpath_list = (
                    xpath_value if isinstance(xpath_value, list) else [xpath_value]
                )

                # 複数のXPATHを試す
                for xpath in xpath_list:
                    try:
                        element = driver.find_element(By.XPATH, xpath)
                        if element:  # 要素が見つかったら終了
                            break
                    except Exception:
                        continue

                if element:
                    # keyが'_link'で終わる場合はhref属性を取得する
                    if key.endswith("_link"):
                        link_value = element.get_attribute("href")
                        setattr(self, key, link_value)
                    # その他の場合はtextを取得する
                    else:
                        text_value = element.text
                        setattr(self, key, text_value)
                else:
                    print(f"{key}に対応する要素が見つかりませんでした。")

            # 数値の属性を正規化
            self.normalize_number()

        except Exception as e:
            print(f"情報取得中にエラーが発生しました: {e}")

    def normalize_number(self) -> None:
        """
        バグ対処の応急処置
        """

        keys = [
            "ranking",
            "comment_count_per_hour",
            "total_comment_count_with_reply",
            "total_comment_count_without_reply",
            "agreements_count",
            "acknowledgements_count",
            "disagreements_count",
            "reply_count",
            "total_comment_count",
            "total_agreements_count",
            "total_acknowledgements_count",
            "total_disagreements_count",
            "learn_count",
            "clarity_count",
            "new_perspective_count",
        ]
        for key in keys:
            # 属性が存在し、Noneではない場合に処理を実行
            if hasattr(self, key) and getattr(self, key) is not None:
                try:
                    value = functions.normalize_number(getattr(self, key))
                    setattr(self, key, value)
                except:
                    pass


class Article(DBBase):
    """
    article_id: int,
    article_link: str,
    article_title: str,
    posted_time: str,
    author: str,
    ranking: int,
    article_genre: str,
    article_content: str,
    comment_count_per_hour: int,
    total_comment_count_with_reply: int,
    total_comment_count_without_reply: int,
    """


class Comment(DBBase):
    """
    comment_id: int,
    article_id: int,
    parent_comment_id: int | None,
    username: str,
    user_link: str,
    posted_time: str,
    comment_content: str,
    agreements_count: int,
    acknowledgements_count: int,
    disagreements_count: int,
    reply_count: int,
    """
