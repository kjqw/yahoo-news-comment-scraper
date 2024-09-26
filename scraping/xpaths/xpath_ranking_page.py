# ページのURL
URL_COMMENT_RANKING = "https://news.yahoo.co.jp/ranking/comment"
URL_ACCESS_RANKING = "https://news.yahoo.co.jp/ranking/access/news"

# ジャンルごとのランキングページはURLで指定できる
# 例: https://news.yahoo.co.jp/ranking/comment/business
URL_GENRES = [
    "",  # 総合
    "domestic",  # 国内
    "world",  # 国際
    "business",  # 経済
    "entertainment",  # エンタメ
    "sports",  # スポーツ
    "it-science",  # IT・科学
    "life",  # ライフ
    "local",  # 地域
]

# 記事のclass属性
CLASS_ARTICLE_LINKS = "newsFeed_item_link"
CLASS_ARTICLE_TITLE = "newsFeed_item_title"

# 記事のXPath
XPATH_ARTICLE_LINKS = "/html/body/div[1]/div/main/div[1]/div/div[2]/ol/li/a"
RELATIVE_XPATH_ARTICLE_TITLE = "div/div[2]/div[1]"
RELATIVE_XPATH_ARTICLE_AUTHOR = "div/div[2]/div[2]/div[1]/span"
RELATIVE_XPATH_ARTICLE_POSTED_TIME = "div/div[2]/div[2]/div[1]/time"
RELATIVE_XPATH_ARTICLE_COMMENT_COUNT_PER_HOUR = "div/div[2]/div[2]/div[2]/p/em"
