from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement


class Article:
    """
    記事の基本情報を格納するクラス。

    Attributes
    ----------
    link : dict[datetime, str] | None
        リンク
    genre : dict[datetime, str] | None
        ジャンル
    title : dict[datetime, str] | None
        タイトル
    author : dict[datetime, str] | None
        著者
    author_link : dict[datetime, str] | None
        著者のリンク
    posted_time : dict[datetime, str] | None
        投稿日時
    updated_time : dict[datetime, str] | None
        更新日時
    ranking : dict[datetime, int] | None
        ランキング
    content : dict[datetime, str] | None
        本文
    comment_count : dict[datetime, int] | None
        コメント数
    comments : list[GeneralComment] | None
        一般コメントのリスト
    expert_comments : list[ExpertComment] | None
        専門家コメントのリスト
    learn_count : dict[datetime, int] | None
        「学びになった」の数
    clarity_count : dict[datetime, int] | None
        「わかりやすい」の数
    new_perspective_count : dict[datetime, int] | None
        「新しい視点」の数
    related_articles : list[Article] | None
        「関連記事」のリスト
    read_also_articles : list[Article] | None
        「あわせて読みたい記事」のリスト
    scraped_time : list[datetime] | None
        スクレイピングされた日時
    """

    def __init__(self):
        self.article_link: dict[datetime, str] | None = None
        self.article_genre: dict[datetime, str] | None = None
        self.article_title: dict[datetime, str] | None = None
        self.author: dict[datetime, str] | None = None
        self.author_link: dict[datetime, str] | None = None
        self.posted_time: dict[datetime, str] | None = None
        self.updated_time: dict[datetime, str] | None = None
        self.ranking: dict[datetime, int] | None = None
        self.content: dict[datetime, str] | None = None
        self.comment_count: dict[datetime, int] | None = None
        self.comments: list[GeneralComment] | None = None
        self.expert_comments: list[ExpertComment] | None = None
        self.learn_count: dict[datetime, int] | None = None
        self.clarity_count: dict[datetime, int] | None = None
        self.new_perspective_count: dict[datetime, int] | None = None
        self.related_articles: list[Article] | None = None
        self.read_also_articles: list[Article] | None = None
        self.scraped_time: list[datetime] | None = None

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
        for key, xpath in xpaths.items():
            try:
                element = driver.find_element(By.XPATH, xpath)
                # scraped_timeがNoneなら空のリストに初期化
                if self.scraped_time is None:
                    self.scraped_time = []
                # 現在の時刻を追加
                self.scraped_time.append(datetime.now())

                # keyが'_link'で終わる場合はhref属性を取得する
                if key.endswith("_link"):
                    link_value = element.get_attribute("href")
                    if getattr(self, key) is None:
                        setattr(self, key, {})
                    getattr(self, key)[self.scraped_time[-1]] = link_value

                # その他のtextデータを時刻とともに辞書に保存する
                else:
                    text_value = element.text
                    if getattr(self, key) is None:
                        setattr(self, key, {})
                    getattr(self, key)[self.scraped_time[-1]] = text_value

            except Exception as e:
                print(f"{key}を取得中にエラーが発生しました: {e}")


class Comment:
    """
    コメントの基本情報を格納するクラス。

    Attributes
    ----------
    article: Article | None
        コメント先の記事
    username : dict[datetime, str] | None
        ユーザ名
    user_link : dict[datetime, str] | None
        ユーザのリンク
    posted_time : dict[datetime, str] | None
        投稿日時
    comment_text : dict[datetime, str] | None
        コメントの本文
    agreements : dict[datetime, int] | None
        「共感した」の数
    acknowledgements : dict[datetime, int] | None
        「参考になった」の数
    disagreements : dict[datetime, int] | None
        「うーん」の数
    scraped_time : list[datetime] | None
        スクレイピングされた日時
    """

    def __init__(self):
        self.article: Article | None = None
        self.username: dict[datetime, str] | None = None
        self.user_link: dict[datetime, str] | None = None
        self.posted_time: dict[datetime, str] | None = None
        self.comment_text: dict[datetime, str] | None = None
        self.agreements: dict[datetime, int] | None = None
        self.acknowledgements: dict[datetime, int] | None = None
        self.disagreements: dict[datetime, int] | None = None
        self.scraped_time: list[datetime] | None = None

    def get_info(self, webelement: WebElement, xpaths: dict[str, str]) -> None:
        """
        渡されたXPATHの辞書を用いてコメントの情報を取得する。
        '_link'で終わるキーの場合はhref属性を、それ以外の場合はtextを取得する。

        Parameters
        ----------
        webelement : WebElement
            コメントの要素
        xpaths : dict[str, str]
            XPATHの辞書
        """
        for key, xpath in xpaths.items():
            try:
                element = webelement.find_element(By.XPATH, xpath)
                # scraped_timeがNoneなら空のリストに初期化
                if self.scraped_time is None:
                    self.scraped_time = []
                # 現在の時刻を追加
                self.scraped_time.append(datetime.now())

                # keyが'_link'で終わる場合はhref属性を取得する
                if key.endswith("_link"):
                    link_value = element.get_attribute("href")
                    if getattr(self, key) is None:
                        setattr(self, key, {})
                    getattr(self, key)[self.scraped_time[-1]] = link_value

                # その他のtextデータを時刻とともに辞書に保存する
                else:
                    text_value = element.text
                    if getattr(self, key) is None:
                        setattr(self, key, {})
                    getattr(self, key)[self.scraped_time[-1]] = text_value

            except Exception as e:
                print(f"{key}を取得中にエラーが発生しました: {e}")


class ExpertComment(Comment):
    """
    専門家コメントの情報を格納するクラス。

    Attributes
    ----------
    expert_type : str | None
        専門家の種類
    """

    def __init__(self):
        # 親クラスの初期化
        super().__init__()
        self.expert_type: str | None = None

        # agreementsとdisagreements属性を削除
        delattr(self, "agreements")
        delattr(self, "disagreements")


class GeneralComment(Comment):
    """
    一般コメントの情報を格納するクラス。

    Attributes
    ----------
    reply_count : int | None
        返信の数
    reply_comments : list[ReplyComment] | None
        返信コメントのリスト
    """

    def __init__(self):
        # 親クラスの初期化
        super().__init__()
        self.reply_count: int | None = None
        self.reply_comments: list[ReplyComment] | None = None


class ReplyComment(Comment):
    """
    返信コメントの情報を格納するクラス。

    Attributes
    ----------
    base_comment : GeneralComment | None
        返信先の一般コメント
    """

    def __init__(self):
        # 親クラスの初期化
        super().__init__()
        self.base_comment: GeneralComment | None = None
