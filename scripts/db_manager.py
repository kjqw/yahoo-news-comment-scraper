import psycopg2

DB_CONFIG = {
    "host": "postgresql_db",
    "database": "yahoo_news",
    "user": "kjqw",
    "password": "1122",
    "port": "5432",
}


def execute_query(
    query: str,
    # params: tuple | None = None,
    db_config: dict = DB_CONFIG,
    commit: bool = False,
) -> list[tuple]:
    """
    指定されたクエリを実行し、結果を取得する関数。

    Parameters
    ----------
    query : str
        実行するクエリ
    commit : bool, Optional
        データ変更が必要な場合にコミットするためのオプション

    Returns
    -------
    list[tuple]
        クエリの結果
    """
    conn = None
    result = []
    try:
        conn = psycopg2.connect(**db_config)
        with conn.cursor() as cur:
            cur.execute(query)
            # cur.execute(query, params)
            if commit:
                conn.commit()
            else:
                result = cur.fetchall()
    except Exception as e:
        if conn:
            conn.rollback()  # エラー発生時にロールバック
        print(f"データベースエラー: {e}")
    finally:
        if conn:
            conn.close()

    return result
