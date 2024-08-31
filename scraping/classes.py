class GeneralComment:
    """
    一般コメントの情報を格納するクラス。

    Attributes
    ----------
    username : str
        ユーザ名
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
    reply_count : int
        返信の数
    reply_comments : list[ReplyComment]
        返信コメントのリスト
    """

    def __init__(
        self,
        username: str,
        posted_time: str,
        comment_text: str,
        agreements: int,
        acknowledgements: int,
        disagreements: int,
        reply_count: int,
    ):
        # コメントの情報
        self.username: str = username
        self.posted_time: str = posted_time
        self.comment_text: str = comment_text
        self.agreements: int = agreements
        self.acknowledgements: int = acknowledgements
        self.disagreements: int = disagreements
        self.reply_count: int = reply_count

        # 返信コメントのリスト
        self.reply_comments: list[ReplyComment] = []


class ReplyComment:
    """
    返信コメントの情報を格納するクラス。

    Attributes
    ----------
    username : str
        ユーザ名
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
    base_comment : GeneralComment | None
        返信先の一般コメント
    """

    def __init__(
        self,
        username: str,
        posted_time: str,
        comment_text: str,
        agreements: int,
        acknowledgements: int,
        disagreements: int,
    ):
        # コメントの情報
        self.username: str = username
        self.posted_time: str = posted_time
        self.comment_text: str = comment_text
        self.agreements: int = agreements
        self.acknowledgements: int = acknowledgements
        self.disagreements: int = disagreements

        # 返信先のコメント
        self.base_comment: GeneralComment | None = None

    def set_base_comment(self, base_comment: GeneralComment) -> None:
        self.base_comment = base_comment
