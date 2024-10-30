from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor

# データベースへの接続情報
DB_INFO = {
    "dbname": "yahoo_news",
    "user": "kjqw",
    "password": "1122",
    "host": "postgresql_db",
    "port": "5432",
}


def create_connection():
    """データベース接続を確立する"""
    conn = psycopg2.connect(**DB_INFO)
    return conn


def add_version() -> int:
    """
    新しいバージョンを追加し、そのIDを返す

    Returns
    -------
    int
        新しいバージョンID
    """
    conn = create_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO versions DEFAULT VALUES RETURNING version_id;")
            version_id = cur.fetchone()[0]
    conn.close()
    return version_id


def insert_article(
    link: str,
    title: str,
    author: str,
    posted_time: datetime,
    ranking: int,
    comments_count: int,
    version_id: int,
) -> None:
    """
    記事情報をarticlesテーブルに挿入する

    Parameters
    ----------
    link : str
        記事リンク
    title : str
        記事タイトル
    author : str
        記事の著者
    posted_time : datetime
        記事の投稿日
    ranking : int
        記事のランキング
    comments_count : int
        記事の1時間当たりのコメント数
    version_id : int
        現在のバージョンID
    """
    conn = create_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO articles (link, title, author, posted_time, ranking, comments_count_per_hour, version_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """,
                (link, title, author, posted_time, ranking, comments_count, version_id),
            )
    conn.close()


def log_error(
    error_message: str, function_name: str, article_id: int | None = None
) -> None:
    """
    エラーメッセージをerrorsテーブルに記録する

    Parameters
    ----------
    error_message : str
        エラーメッセージ
    function_name : str
        エラーが発生した関数名
    article_id : int | None, optional
        関連する記事のID。デフォルトはNone。
    """
    conn = create_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO errors (error_message, function_name, article_id)
                VALUES (%s, %s, %s);
            """,
                (error_message, function_name, article_id),
            )
    conn.close()


# デモデータの挿入
if __name__ == "__main__":
    try:
        # 新しいバージョンを生成
        version_id = add_version()
        print(f"新しいバージョンID: {version_id}")

        # ダミー記事の挿入
        insert_article(
            link="https://example.com/article/123",
            title="サンプル記事",
            author="著者名",
            posted_time=datetime.now(),
            ranking=1,
            comments_count=5,
            version_id=version_id,
        )
        print("記事が挿入されました。")

    except Exception as e:
        # エラーログの記録
        log_error(str(e), "main")
        print("エラーが発生しました。エラーログが記録されました。")
