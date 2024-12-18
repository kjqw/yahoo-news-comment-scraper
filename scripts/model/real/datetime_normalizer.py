# %%
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from db_manager import execute_query

# %%
db_config = {
    "host": "postgresql_db",
    "database": "yahoo_news",
    "user": "kjqw",
    "password": "1122",
    "port": "5432",
}

# %%
scraped_times = execute_query(
    """
    SELECT scraped_time
    FROM comments
    """
)
scraped_times = [i[0] for i in scraped_times]

posted_times = execute_query(
    """
    SELECT posted_time
    FROM comments
    """
)
posted_times = [i[0] for i in posted_times]

# %%
# scraped_times
posted_times
# %%
# len(scraped_times)
len(posted_times)


# %%
def normalize_time(posted_time: str, scraped_time: datetime) -> datetime:
    """
    時間表記を正規化し、datetimeオブジェクトとして返す（分までの精度）.

    Parameters
    ----------
    posted_time : str
        投稿時間の表記（例: "1時間前", "たった今", "10/10(木) 13:11", "2022/9/10(土) 13:01"など）
    scraped_time : datetime
        スクレイピング時の基準時間

    Returns
    -------
    datetime
        正規化された投稿時間（秒以下切り捨て）
    """
    # "たった今" の処理
    if posted_time == "たった今":
        return scraped_time.replace(second=0, microsecond=0)

    # "~分前", "~時間前", "~日前" の処理
    if "前" in posted_time:
        match = re.match(r"(\d+)(分|時間|日)前", posted_time)
        if not match:
            raise ValueError(f"正規表現にマッチしません: {posted_time}")
        value = int(match.group(1))
        unit = match.group(2)
        if unit == "分":
            result = scraped_time - timedelta(minutes=value)
        elif unit == "時間":
            result = scraped_time - timedelta(hours=value)
        elif unit == "日":
            result = scraped_time - timedelta(days=value)
        else:
            raise ValueError(f"未対応の単位: {unit}")
        return result.replace(second=0, microsecond=0)

    # "mm/dd(曜日) hh:MM" または "YYYY/mm/dd(曜日) hh:MM" の処理
    match = re.match(
        r"(?:(\d{4})/)?(\d{1,2})/(\d{1,2})\(.\)\s(\d{1,2}):(\d{1,2})", posted_time
    )
    if match:
        year = int(match.group(1)) if match.group(1) else scraped_time.year
        month = int(match.group(2))
        day = int(match.group(3))
        hour = int(match.group(4))
        minute = int(match.group(5))

        # 年が指定されていない場合、未来の日付を除外するため年度を調整
        if (
            not match.group(1)
            and datetime(year, month, day, hour, minute) > scraped_time
        ):
            year -= 1

        return datetime(year, month, day, hour, minute)

    # 該当しないフォーマットは例外を発生
    raise ValueError(f"未対応のフォーマット: {posted_time}")


# 動作確認用コード
if __name__ == "__main__":
    examples = [
        ("たった今", datetime.now()),
        ("1時間前", datetime(2024, 12, 11, 21, 51, 14, 373321)),
        ("26分前", datetime(2022, 2, 3, 5, 31, 49, 846290)),
        ("1日前", datetime(2023, 1, 1, 0, 0, 0, 0)),
        ("10/10(木) 13:11", datetime(2023, 10, 10, 13, 11, 0, 0)),
        ("2022/9/10(土) 13:01", datetime(2022, 9, 10, 13, 0, 0, 0)),
    ]

    for posted_time, scraped_time in examples:
        try:
            normalized_time = normalize_time(posted_time, scraped_time)
            print(f"posted_time: {posted_time}, normalized_time: {normalized_time}")
        except ValueError as e:
            print(e)

# %%
results = []
for posted_time, scraped_time in zip(posted_times, scraped_times):
    try:
        results.append(normalize_time(posted_time, scraped_time))
        print(posted_time, scraped_time)
    except:
        # print(posted_time, scraped_time)
        pass
# %%
sorted(results)
# %%
len(results)
# %%
