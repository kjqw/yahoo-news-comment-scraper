# 記事ページの要素
XPATH_ARTICLE_TITLE = "/html/body/div[1]/div/main/div[1]/div[1]/article/header/h1/a"
XPATH_ARTICLE_AUTHOR = "/html/body/div[1]/div/main/div[1]/div[1]/article/header/p/a"
XPATH_ARTICLE_POSTED_TIME = (
    "/html/body/div[1]/div/main/div[1]/div[1]/article/header/p/span/time"
)

# コメント数
# XPATH_TOTAL_COMMENT_COUNT_WITH_REPLY = (
#     "/html/body/div[1]/div/main/div[1]/div[1]/article/div[2]/div/p/span"
# )
# XPATH_TOTAL_COMMENT_COUNT_WITHOUT_REPLY = (
#     "/html/body/div[1]/div/main/div[1]/div[1]/article/div[4]/div/p/span"
# )
XPATH_TOTAL_COMMENT_COUNT_WITH_REPLY = (
    "/html/body/div[1]/div/main/div[1]/div[1]/article/div[4]/div/p/span"
)
XPATH_TOTAL_COMMENT_COUNT_WITHOUT_REPLY = (
    "/html/body/div[1]/div/main/div[1]/div[1]/article/div[6]/div/p/span"
)

# 専門家コメントの要素
XPATH_EXPERT_COMMENT_SECTIONS = (
    "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li"
)
RELATIVE_XPATH_EXPERT_USERNAME = "article/div[2]/div[1]/h2/a"
RELATIVE_XPATH_EXPERT_POSTED_TIME = "article/div[2]/div[1]/time/a"
RELATIVE_XPATH_EXPERT_TYPE = "article/div[2]/div[1]/p"
RELATIVE_XPATH_EXPERT_COMMENT_BUTTON = "article/div[2]/p/button"
RELATIVE_XPATH_EXPERT_COMMENT_TEXT = "article/div[2]/p"
RELATIVE_XPATH_EXPERT_AGREEMENTS = "article/div[2]/div[2]/button/span"

# 一般コメントの要素
XPATH_GENERAL_COMMENT_SECTIONS = (
    "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li"
)
RELATIVE_XPATH_GENERAL_COMMENT_USERNAME = "article/article/div[2]/div[1]/h2/a"
RELATIVE_XPATH_GENERAL_COMMENT_POSTED_TIME = "article/article/div[2]/div[1]/time/a"
RELATIVE_XPATH_GENERAL_COMMENT_COMMENT_TEXT = "article/article/div[2]/p"
RELATIVE_XPATH_GENERAL_COMMENT_AGREEMENTS = (
    "article/article/div[2]/div[2]/div/ul/li[1]/button[2]/span/span[2]"
)
RELATIVE_XPATH_GENERAL_COMMENT_ACKNOWLEDGEMENTS = (
    "article/article/div[2]/div[2]/div/ul/li[2]/button[2]/span/span[2]"
)
RELATIVE_XPATH_GENERAL_COMMENT_DISAGREEMENTS = (
    "article/article/div[2]/div[2]/div/ul/li[3]/button[2]/span/span[2]"
)
RELATIVE_XPATH_GENERAL_COMMENT_REPLY_COUNT = (
    "article/article/div[2]/div[2]/button[1]/span"
)
RELATIVE_XPATH_GENERAL_COMMENT_REPLY_BUTTON = "article/article/div[2]/div[2]/button[1]"
RELATIVE_XPATH_GENERAL_COMMENT_REPLY_BUTTON_MORE = "div/div[1]/div/button"

# 返信コメントの要素
XPATH_REPLY_COMMENT_SECTIONS = f"{XPATH_GENERAL_COMMENT_SECTIONS}/div/ul/li"
RELATIVE_XPATH_REPLY_COMMENT_USERNAME = "div/article/div[2]/div[1]/h2/a"
RELATIVE_XPATH_REPLY_COMMENT_POSTED_TIME = "div/article/div[2]/div[1]/time"
RELATIVE_XPATH_REPLY_COMMENT_COMMENT_TEXT = "div/article/div[2]/p"
RELATIVE_XPATH_REPLY_COMMENT_AGREEMENTS = (
    "div/article/div[2]/div[2]/ul/li[1]/button[2]/span/span[2]"
)
RELATIVE_XPATH_REPLY_COMMENT_ACKNOWLEDGEMENTS = (
    "div/article/div[2]/div[2]/ul/li[2]/button[2]/span/span[2]"
)
RELATIVE_XPATH_REPLY_COMMENT_DISAGREEMENTS = (
    "div/article/div[2]/div[2]/ul/li[3]/button[2]/span/span[2]"
)
