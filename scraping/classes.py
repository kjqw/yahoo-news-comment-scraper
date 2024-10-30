import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement


class Savable(ABC):
    """
    データを保存できるクラスの基底クラス。
    """

    @abstractmethod
    def export_data(self) -> dict:
        """
        jsonに書き込むためのデータを返す。具体的なクラスで実装する必要がある。
        """
        pass

    def save_data(self, save_path: Path) -> None:
        """
        jsonに書き込むためのデータを保存する。既存のjsonファイルがある場合はデータを追加する。

        Parameters
        ----------
        save_path : Path
            保存先のパス
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 既存のデータを読み込む（ファイルが存在する場合）
        if save_path.exists():
            with open(save_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        # 新しいデータをエクスポート
        new_data = self.export_data()

        # 既存のデータと新しいデータをマージ
        existing_data.update(new_data)

        # データをファイルに書き込む
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

    def _serialize(self, obj) -> str | int | dict:
        """
        オブジェクトがArticleやCommentクラスであればidを、それ以外の場合はそのまま返す。

        Parameters
        ----------
        obj : Any
            シリアライズ対象のオブジェクト

        Returns
        -------
        str | int | dict
            シリアライズされた結果
        """
        if isinstance(
            obj,
            (
                Article,
                Comment,
                ExpertComment,
                GeneralComment,
                ReplyComment,
            ),
        ):
            return id(obj)
        elif isinstance(obj, list):  # listをチェック
            return [
                self._serialize(item) for item in obj
            ]  # リスト内の要素を再帰的にシリアライズ
        elif isinstance(obj, dict):  # dictをチェック
            return {
                key: self._serialize(value) for key, value in obj.items()
            }  # 辞書のキー・値を再帰的にシリアライズ
        elif isinstance(obj, datetime):  # datetimeをチェック
            return obj.isoformat()  # datetimeをISOフォーマットの文字列に変換
        return obj

    def export_data(self) -> dict:
        """
        オブジェクトのすべての属性を動的にシリアライズして辞書形式で返す。

        Returns
        -------
        dict
            シリアライズされたオブジェクトのデータ
        """
        data = {}
        for key, value in self.__dict__.items():
            data[key] = self._serialize(value)
        return {id(self): data}


class Article(Savable):
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
    comment_count_per_hour : dict[int, int] | None
        1時間あたりのコメント数
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
        self.article_link: dict[int, str] | None = {}
        self.article_genre: dict[int, str] | None = {}
        self.article_title: dict[int, str] | None = {}
        self.author: dict[int, str] | None = {}
        self.author_link: dict[int, str] | None = {}
        self.posted_time: dict[int, str] | None = {}
        self.updated_time: dict[int, str] | None = {}
        self.ranking: dict[int, int] | None = {}
        self.content: dict[int, str] | None = {}
        self.comment_count: dict[int, int] | None = {}
        self.comment_count_per_hour: dict[int, int] | None = {}
        self.comments: list[GeneralComment] | None = []
        self.expert_comments: list[ExpertComment] | None = []
        self.learn_count: dict[int, int] | None = {}
        self.clarity_count: dict[int, int] | None = {}
        self.new_perspective_count: dict[int, int] | None = {}
        self.related_articles: list[Article] | None = []
        self.read_also_articles: list[Article] | None = []
        self.scraped_time: dict[int, datetime] | None = {}
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
        self.scraped_time[self.scraping_count + 1] = datetime.now()

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
                        getattr(self, key)[self.scraping_count + 1] = link_value

                    # その他のtextデータを時刻とともに辞書に保存する
                    else:
                        text_value = element.text
                        if getattr(self, key) is None:
                            setattr(self, key, {})
                        getattr(self, key)[self.scraping_count + 1] = text_value

                except Exception as e:
                    print(f"{key}を取得中にエラーが発生しました: {e}")

            # スクレイピング回数を更新
            self.scraping_count += 1

        except:
            pass


class Comment(Savable):
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
        self.username: dict[int, str] | None = {}
        self.user_link: dict[int, str] | None = {}
        self.posted_time: dict[int, str] | None = {}
        self.comment_text: dict[int, str] | None = {}
        self.agreements: dict[int, int] | None = {}
        self.acknowledgements: dict[int, int] | None = {}
        self.disagreements: dict[int, int] | None = {}
        self.scraped_time: dict[int, datetime] | None = {}
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
        self.scraped_time[self.scraping_count + 1] = datetime.now()

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
                        getattr(self, key)[self.scraping_count + 1] = link_value

                    # その他のtextデータを時刻とともに辞書に保存する
                    else:
                        text_value = element.text
                        if getattr(self, key) is None:
                            setattr(self, key, {})
                        getattr(self, key)[self.scraping_count + 1] = text_value

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
        self.reply_comments: list[ReplyComment] | None = []


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
