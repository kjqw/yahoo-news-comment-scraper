# ユーザーの情報のXPath
XPATH_USER_NAME = "/html/body/div[1]/div/main/div[1]/div/div[1]/div/div[2]/p[1]"
XPATH_POSTED_COMMENT_COUNT = "/html/body/div[1]/div/main/div[1]/div/div[2]/div/span[2]"
XPATH_TOTAL_AGREEMENTS = (
    "/html/body/div[1]/div/main/div[1]/div/div[2]/ul/li[1]/span/span[2]"
)
XPATH_TOTAL_ACKNOWLEDGEMENTS = (
    "/html/body/div[1]/div/main/div[1]/div/div[2]/ul/li[2]/span/span[2]"
)
XPATH_TOTAL_DISAGREEMENTS = (
    "/html/body/div[1]/div/main/div[1]/div/div[2]/ul/li[3]/span/span[2]"
)

# コメント部分のXPath
XPATH_COMMENT_SECTIONS = "/html/body/div[1]/div/main/div[1]/div/div[3]/ul/li[i1]"

RELATIVE_XPATH_ARTICLE_LINK = "article[1]/a"
RELATIVE_XPATH_ARTICLE_TITLE = "article[1]/a/div/h2/p[1]"
RELATIVE_XPATH_ARTICLE_AUTHOR = "article[1]/a/div/h2/p[2]/span"
RELATIVE_XPATH_ARTICLE_POSTED_TIME = "article[1]/a/div/h2/p[2]/time"
RELATIVE_XPATH_ARTICLE_COMMENT_PAGE_LINK = "article[1]/div/a"

RELATIVE_XPATH_COMMENT_POSTED_TIME = "article[2]/div/div/div[1]/a"
RELATIVE_XPATH_COMMENT_TEXT = "article[2]/div/div/div[2]/p"
RELATIVE_XPATH_COMMENT_AGREEMENTS = (
    "article[2]/div/div/div[2]/div/ul/li[1]/div/span/span[2]"
)
RELATIVE_XPATH_COMMENT_ACKNOWLEDGEMENTS = (
    "article[2]/div/div/div[2]/div/ul/li[2]/div/span/span[2]"
)
RELATIVE_XPATH_COMMENT_DISAGREEMENTS = (
    "article[2]/div/div/div[2]/div/ul/li[3]/div/span/span[2]"
)
RELATIVE_XPATH_COMMENT_REPLY_COUNT = (
    "article[2]/div/div/div[2]/div/a/span[2]"  # ないこともある
)
RELATIVE_XPATH_COMMENT_REPLY_COMMENT = "article[2]/div/div[1]/a"  # ないこともある
