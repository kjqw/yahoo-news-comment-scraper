from selenium import webdriver
from selenium.webdriver.common.by import By


class Article:
    """
    記事の基本情報を格納するクラス。

    Attributes
    ----------
    link : str
        リンク
    genre : str
        ジャンル
    title : str
        タイトル
    author : str
        著者
    author_link : str
        著者のリンク
    posted_time : str
        投稿日時
    updated_time : str
        更新日時
    content : str
        本文
    comment_count : int
        コメント数
    comments : list[GeneralComment]
        一般コメントのリスト
    expert_comments : list[ExpertComment]
        専門家コメントのリスト
    learn_count : int
        「学びになった」の数
    clarity_count : int
        「わかりやすい」の数
    new_perspective_count : int
        「新しい視点」の数
    related_articles : list[Article]
        「関連記事」のリスト
    read_also_articles : list[Article]
        「あわせて読みたい記事」のリスト
    """

    def __init__(self):
        # 初期化時は空の状態にしておく
        self.article_link: str = ""
        self.article_genre: str = ""
        self.article_title: str = ""
        self.author: str = ""
        self.author_link: str = ""
        self.posted_time: str = ""
        self.updated_time: str = ""
        self.content: str = ""
        self.comment_count: int = 0
        self.comments: list[GeneralComment] = []
        self.expert_comments: list[ExpertComment] = []
        self.learn_count: int = 0
        self.clarity_count: int = 0
        self.new_perspective_count: int = 0
        self.related_articles: list[Article] = []
        self.read_also_articles: list[Article] = []

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
                # keyが'_link'で終わる場合はhref属性を取得する
                if key.endswith("_link"):
                    setattr(self, key, element.get_attribute("href"))
                # それ以外ならtextを取得する
                else:
                    setattr(self, key, element.text)
            except Exception as e:
                print(f"{key}を取得中にエラーが発生しました: {e}")


class Comment:
    """
    コメントの基本情報を格納するクラス。

    Attributes
    ----------
    article: Article | None
        コメント先の記事
    username : str
        ユーザ名
    user_link : str
        ユーザのリンク
    posted_time : str
        投稿日時
    comment_text : str
        コメントの本文
    agreements : int
        「共感した」の数
    acknowledgements : int
        「参考になった」の数
    disagreements : int
        「うーん」の数
    """

    def __init__(self):
        # 初期化時は空の状態にしておく
        self.article: Article | None = None
        self.username: str = ""
        self.user_link: str = ""
        self.posted_time: str = ""
        self.comment_text: str = ""
        self.agreements: int = 0
        self.acknowledgements: int = 0
        self.disagreements: int = 0

    def get_info(self, driver: webdriver, xpaths: dict[str, str]) -> None:
        """
        渡されたXPATHの辞書を用いてコメントの情報を取得する。
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
                # keyが'_link'で終わる場合はhref属性を取得する
                if key.endswith("_link"):
                    setattr(self, key, element.get_attribute("href"))
                # それ以外ならtextを取得する
                else:
                    setattr(self, key, element.text)
            except Exception as e:
                print(f"{key}を取得中にエラーが発生しました: {e}")


class ExpertComment(Comment):
    """
    専門家コメントの情報を格納するクラス。

    Attributes
    ----------
    expert_type : str
        専門家の種類
    """

    def __init__(self):
        # 親クラスの初期化
        super().__init__()
        self.expert_type: str = ""

        # agreementsとdisagreements属性を削除
        delattr(self, "agreements")
        delattr(self, "disagreements")


class GeneralComment(Comment):
    """
    一般コメントの情報を格納するクラス。

    Attributes
    ----------
    reply_count : int
        返信の数
    reply_comments : list[ReplyComment]
        返信コメントのリスト
    """

    def __init__(self):
        # 親クラスの初期化
        super().__init__()
        self.reply_count: int = 0
        self.reply_comments: list[ReplyComment] = []


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
