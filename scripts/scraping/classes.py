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

    def get_info(self, driver: webdriver, xpaths: dict[str, str]) -> None:
        """
        渡されたXPATHの辞書を用いて記事の情報を取得する。
        '_link'で終わるキーの場合はhref属性を、それ以外の場合はtextを取得する。

        Parameters
        ----------
        driver : webdriver
            WebDriverインスタンス
        xpaths : dict[str, str]
            XPATHの辞書
        """
        try:
            # 各XPATHを用いて情報を取得
            for key, xpath in xpaths.items():
                try:
                    element = driver.find_element(By.XPATH, xpath)

                    # keyが'_link'で終わる場合はhref属性を取得する
                    if key.endswith("_link"):
                        link_value = element.get_attribute("href")
                        setattr(self, key, link_value)
                    # その他のtextデータを取得する
                    else:
                        text_value = element.text
                        setattr(self, key, text_value)

                except Exception as e:
                    print(f"{key}を取得中にエラーが発生しました: {e}")

        except:
            pass


class Article(DBBase):
    """
    article_id: int,
    article_link: str,
    title: str,
    posted_time: str,
    author: str,
    ranking: int,
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
    content: str,
    agreements_count: int,
    acknowledgements_count: int,
    disagreements_count: int,
    reply_count: int,
    """


# if __name__ == "__main__":
#     # 使用例
#     DB_CONFIG = {
#         "host": "postgresql_db",
#         "database": "yahoo_news",
#         "user": "kjqw",
#         "password": "1122",
#         "port": "5432",
#     }

#     article = Article(
#         article_id=1,
#         article_link="https://example.com/article1",
#         title="Example Article",
#         author="Author Name",
#         posted_time="2024-10-30 15:00:00",
#         ranking=1,
#     )

#     comment = Comment(
#         article_id=1,
#         parent_comment_id=None,
#         username="user123",
#         user_link="https://example.com/user123",
#         posted_time="2024-10-30 15:30:00",
#         content="This is a sample comment.",
#         agreements_count=10,
#         acknowledgements_count=2,
#         disagreements_count=1,
#         reply_count=0,
#     )

#     # データベースに保存
#     article.save_data(DB_CONFIG, "articles")
#     comment.save_data(DB_CONFIG, "comments")


# class Article(DBBase():):
#     """
#     記事の基本情報を格納するクラス。

#     Attributes
#     ----------
#     link : str | None
#         リンク
#     genre : str | None
#         ジャンル
#     title : str | None
#         タイトル
#     author : str | None
#         著者
#     author_link : str | None
#         著者のリンク
#     posted_time : str | None
#         投稿日時
#     updated_time : str | None
#         更新日時
#     ranking : int | None
#         ランキング
#     content : str | None
#         本文
#     comment_count : int | None
#         コメント数
#     comment_count_per_hour : int | None
#         1時間あたりのコメント数
#     comments : list[GeneralComment] | None
#         一般コメントのリスト
#     expert_comments : list[ExpertComment] | None
#         専門家コメントのリスト
#     learn_count : int | None
#         「学びになった」の数
#     clarity_count : int | None
#         「わかりやすい」の数
#     new_perspective_count : int | None
#         「新しい視点」の数
#     related_articles : list[Article] | None
#         「関連記事」のリスト
#     read_also_articles : list[Article] | None
#         「あわせて読みたい記事」のリスト
#     scraped_time : datetime | None
#         スクレイピングされた日時
#     """

#     def __init__(self):
#         self.article_link: dict[int, str] | None = {}
#         self.article_genre: dict[int, str] | None = {}
#         self.article_title: dict[int, str] | None = {}
#         self.author: dict[int, str] | None = {}
#         self.author_link: dict[int, str] | None = {}
#         self.posted_time: dict[int, str] | None = {}
#         self.updated_time: dict[int, str] | None = {}
#         self.ranking: dict[int, int] | None = {}
#         self.content: dict[int, str] | None = {}
#         self.comment_count: dict[int, int] | None = {}
#         self.comment_count_per_hour: dict[int, int] | None = {}
#         self.comments: list[GeneralComment] | None = []
#         self.expert_comments: list[ExpertComment] | None = []
#         self.learn_count: dict[int, int] | None = {}
#         self.clarity_count: dict[int, int] | None = {}
#         self.new_perspective_count: dict[int, int] | None = {}
#         self.related_articles: list[Article] | None = []
#         self.read_also_articles: list[Article] | None = []
#         self.scraped_time: dict[int, datetime] | None = {}
#         self.scraping_count: int = 0

#     def get_info(self, driver: webdriver, xpaths: dict[str, str]) -> None:
#         """
#         渡されたXPATHの辞書を用いて記事の情報を取得する。
#         '_link'で終わるキーの場合はhref属性を、それ以外の場合はtextを取得する。

#         Parameters
#         ----------
#         driver : webdriver
#             WebDriverインスタンス
#         xpaths : dict[str, str]
#             XPATHの辞書
#         """
#         # scraped_timeがNoneなら空の辞書に初期化
#         if self.scraped_time is None:
#             self.scraped_time = {}
#         # 現在の時刻を追加
#         self.scraped_time[self.scraping_count + 1] = datetime.now()

#         try:
#             # 各XPATHを用いて情報を取得
#             for key, xpath in xpaths.items():
#                 try:
#                     element = driver.find_element(By.XPATH, xpath)

#                     # keyが'_link'で終わる場合はhref属性を取得する
#                     if key.endswith("_link"):
#                         link_value = element.get_attribute("href")
#                         if getattr(self, key) is None:
#                             setattr(self, key, {})
#                         getattr(self, key)[self.scraping_count + 1] = link_value

#                     # その他のtextデータを時刻とともに辞書に保存する
#                     else:
#                         text_value = element.text
#                         if getattr(self, key) is None:
#                             setattr(self, key, {})
#                         getattr(self, key)[self.scraping_count + 1] = text_value

#                 except Exception as e:
#                     print(f"{key}を取得中にエラーが発生しました: {e}")

#         except:
#             pass


# class Comment(DBBase():):
#     """
#     コメントの基本情報を格納するクラス。

#     Attributes
#     ----------
#     article: Article | None
#         コメント先の記事
#     username : dict[int, str] | None
#         ユーザ名
#     user_link : dict[int, str] | None
#         ユーザのリンク
#     posted_time : dict[int, str] | None
#         投稿日時
#     comment_text : dict[int, str] | None
#         コメントの本文
#     agreements : dict[int, int] | None
#         「共感した」の数
#     acknowledgements : dict[int, int] | None
#         「参考になった」の数
#     disagreements : dict[int, int] | None
#         「うーん」の数
#     scraped_time : dict[int, datetime] | None
#         スクレイピングされた日時
#     """

#     def __init__(self):
#         self.article: Article | None = None
#         self.username: dict[int, str] | None = {}
#         self.user_link: dict[int, str] | None = {}
#         self.posted_time: dict[int, str] | None = {}
#         self.comment_text: dict[int, str] | None = {}
#         self.agreements: dict[int, int] | None = {}
#         self.acknowledgements: dict[int, int] | None = {}
#         self.disagreements: dict[int, int] | None = {}
#         self.scraped_time: dict[int, datetime] | None = {}
#         self.scraping_count: int = 0

#     def get_info(self, webelement: WebElement, xpaths: dict[str, str]) -> None:
#         """
#         渡されたXPATHの辞書を用いてコメントの情報を取得する。
#         '_link'で終わるキーの場合はhref属性を、それ以外の場合はtextを取得する。

#         Parameters
#         ----------
#         webelement : WebElement
#             コメントの要素
#         xpaths : dict[str, str]
#             XPATHの辞書
#         """
#         # scraped_timeがNoneなら空の辞書に初期化
#         if self.scraped_time is None:
#             self.scraped_time = {}
#         # 現在の時刻を追加
#         self.scraped_time[self.scraping_count + 1] = datetime.now()

#         try:
#             # 各XPATHを用いて情報を取得
#             for key, xpath in xpaths.items():
#                 try:
#                     element = webelement.find_element(By.XPATH, xpath)

#                     # keyが'_link'で終わる場合はhref属性を取得する
#                     if key.endswith("_link"):
#                         link_value = element.get_attribute("href")
#                         if getattr(self, key) is None:
#                             setattr(self, key, {})
#                         getattr(self, key)[self.scraping_count + 1] = link_value

#                     # その他のtextデータを時刻とともに辞書に保存する
#                     else:
#                         text_value = element.text
#                         if getattr(self, key) is None:
#                             setattr(self, key, {})
#                         getattr(self, key)[self.scraping_count + 1] = text_value

#                 except Exception as e:
#                     print(f"{key}を取得中にエラーが発生しました: {e}")

#         except:
#             pass


# class ExpertComment(Comment):
#     """
#     専門家コメントの情報を格納するクラス。

#     Attributes
#     ----------
#     expert_type : str | None
#         専門家の種類
#     """

#     def __init__(self):
#         # 親クラスの初期化
#         super().__init__()
#         self.expert_type: str | None = None

#         # agreementsとdisagreements属性を削除
#         delattr(self, "agreements")
#         delattr(self, "disagreements")


# class GeneralComment(Comment):
#     """
#     一般コメントの情報を格納するクラス。

#     Attributes
#     ----------
#     reply_count : int | None
#         返信の数
#     reply_comments : list[ReplyComment] | None
#         返信コメントのリスト
#     """

#     def __init__(self):
#         # 親クラスの初期化
#         super().__init__()
#         self.reply_count: int | None = None
#         self.reply_comments: list[ReplyComment] | None = []


# class ReplyComment(Comment):
#     """
#     返信コメントの情報を格納するクラス。

#     Attributes
#     ----------
#     base_comment : GeneralComment | None
#         返信先の一般コメント
#     """

#     def __init__(self):
#         # 親クラスの初期化
#         super().__init__()
#         self.base_comment: GeneralComment | None = None
