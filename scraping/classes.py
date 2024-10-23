from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement


class Article:
    """
    記事の基本情報を格納するクラス。

    Attributes
    ----------
    link : dict[int, str] | None
        リンク
    genre : dict[int, str] | None
        ジャンル
    title : dict[int, str] | None
        タイトル
    author : dict[int, str] | None
        著者
    author_link : dict[int, str] | None
        著者のリンク
    posted_time : dict[int, str] | None
        投稿日時
    updated_time : dict[int, str] | None
        更新日時
    ranking : dict[int, int] | None
        ランキング
    content : dict[int, str] | None
        本文
    comment_count : dict[int, int] | None
        コメント数
    comments : list[GeneralComment] | None
        一般コメントのリスト
    expert_comments : list[ExpertComment] | None
        専門家コメントのリスト
    learn_count : dict[int, int] | None
        「学びになった」の数
    clarity_count : dict[int, int] | None
        「わかりやすい」の数
    new_perspective_count : dict[int, int] | None
        「新しい視点」の数
    related_articles : list[Article] | None
        「関連記事」のリスト
    read_also_articles : list[Article] | None
        「あわせて読みたい記事」のリスト
    scraped_time : list[datetime] | None
        スクレイピングされた日時
    scraping_count : int
        スクレイピング回数
    """

    def __init__(self):
        self.article_link: dict[int, str] | None = None
        self.article_genre: dict[int, str] | None = None
        self.article_title: dict[int, str] | None = None
        self.author: dict[int, str] | None = None
        self.author_link: dict[int, str] | None = None
        self.posted_time: dict[int, str] | None = None
        self.updated_time: dict[int, str] | None = None
        self.ranking: dict[int, int] | None = None
        self.content: dict[int, str] | None = None
        self.comment_count: dict[int, int] | None = None
        self.comments: list[GeneralComment] | None = None
        self.expert_comments: list[ExpertComment] | None = None
        self.learn_count: dict[int, int] | None = None
        self.clarity_count: dict[int, int] | None = None
        self.new_perspective_count: dict[int, int] | None = None
        self.related_articles: list[Article] | None = None
        self.read_also_articles: list[Article] | None = None
        self.scraped_time: dict[int, datetime] | None = None
        self.scraping_count: int = 0

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
        # scraped_timeがNoneなら空の辞書に初期化
        if self.scraped_time is None:
            self.scraped_time = {}
        # 現在の時刻を追加
        self.scraped_time[self.scraping_count] = datetime.now()

        try:
            # 各XPATHを用いて情報を取得
            for key, xpath in xpaths.items():
                try:
                    element = driver.find_element(By.XPATH, xpath)

                    # keyが'_link'で終わる場合はhref属性を取得する
                    if key.endswith("_link"):
                        link_value = element.get_attribute("href")
                        if getattr(self, key) is None:
                            setattr(self, key, {})
                        getattr(self, key)[self.scraping_count] = link_value

                    # その他のtextデータを時刻とともに辞書に保存する
                    else:
                        text_value = element.text
                        if getattr(self, key) is None:
                            setattr(self, key, {})
                        getattr(self, key)[self.scraping_count] = text_value

                except Exception as e:
                    print(f"{key}を取得中にエラーが発生しました: {e}")

            # スクレイピング回数を更新
            self.scraping_count += 1

        except:
            pass


class Comment:
    """
    コメントの基本情報を格納するクラス。

    Attributes
    ----------
    article: Article | None
        コメント先の記事
    username : dict[int, str] | None
        ユーザ名
    user_link : dict[int, str] | None
        ユーザのリンク
    posted_time : dict[int, str] | None
        投稿日時
    comment_text : dict[int, str] | None
        コメントの本文
    agreements : dict[int, int] | None
        「共感した」の数
    acknowledgements : dict[int, int] | None
        「参考になった」の数
    disagreements : dict[int, int] | None
        「うーん」の数
    scraped_time : dict[int, datetime] | None
        スクレイピングされた日時
    scraping_count : int
        スクレイピング回数
    """

    def __init__(self):
        self.article: Article | None = None
        self.username: dict[int, str] | None = None
        self.user_link: dict[int, str] | None = None
        self.posted_time: dict[int, str] | None = None
        self.comment_text: dict[int, str] | None = None
        self.agreements: dict[int, int] | None = None
        self.acknowledgements: dict[int, int] | None = None
        self.disagreements: dict[int, int] | None = None
        self.scraped_time: dict[int, datetime] | None = None
        self.scraping_count: int = 0

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
        # scraped_timeがNoneなら空の辞書に初期化
        if self.scraped_time is None:
            self.scraped_time = {}
        # 現在の時刻を追加
        self.scraped_time[self.scraping_count] = datetime.now()

        try:
            # 各XPATHを用いて情報を取得
            for key, xpath in xpaths.items():
                try:
                    element = webelement.find_element(By.XPATH, xpath)

                    # keyが'_link'で終わる場合はhref属性を取得する
                    if key.endswith("_link"):
                        link_value = element.get_attribute("href")
                        if getattr(self, key) is None:
                            setattr(self, key, {})
                        getattr(self, key)[self.scraping_count] = link_value

                    # その他のtextデータを時刻とともに辞書に保存する
                    else:
                        text_value = element.text
                        if getattr(self, key) is None:
                            setattr(self, key, {})
                        getattr(self, key)[self.scraping_count] = text_value

                except Exception as e:
                    print(f"{key}を取得中にエラーが発生しました: {e}")

            # スクレイピング回数を更新
            self.scraping_count += 1

        except:
            pass


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
