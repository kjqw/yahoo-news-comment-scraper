# 記事ページの要素
XPATH_ARTICLE_TITLE = "/html/body/div[1]/div/main/div[1]/div[1]/article/header/h1/a"
XPATH_ARTICLE_AUTHOR = "/html/body/div[1]/div/main/div[1]/div[1]/article/header/p/a"
XPATH_ARTICLE_POSTED_TIME = (
    "/html/body/div[1]/div/main/div[1]/div[1]/article/header/p/span/time"
)

# コメント数
XPATH_TOTAL_COMMENT_COUNT_WITH_REPLY = (
    "/html/body/div[1]/div/main/div[1]/div[1]/article/div[2]/div/p/span"
)
XPATH_TOTAL_COMMENT_COUNT_WITHOUT_REPLY = (
    "/html/body/div[1]/div/main/div[1]/div[1]/article/div[4]/div/p/span"
)

# 専門家コメントの要素
XPATH_EXPERT_USERNAME = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i1]/article/div[2]/div[1]/h2/a"
XPATH_EXPERT_POSTED_TIME = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i1]/article/div[2]/div[1]/time/a"
XPATH_EXPERT_TYPE = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i1]/article/div[2]/div[1]/p"
XPATH_EXPERT_COMMENT_BUTTON = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i1]/article/div[2]/p/button"
XPATH_EXPERT_COMMENT_TEXT = (
    "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i1]/article/div[2]/p"
)
XPATH_EXPERT_AGREEMENTS = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i1]/article/div[2]/div[2]/button/span"

# 一般コメントの要素
XPATH_GENERAL_COMMENT_SECTIONS = (
    "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]"
)
XPATH_GENERAL_COMMENT_USERNAME = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/article/article/div[2]/div[1]/h2/a"
XPATH_GENERAL_COMMENT_POSTED_TIME = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/article/article/div[2]/div[1]/time/a"
XPATH_GENERAL_COMMENT_COMMENT_TEXT = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/article/article/div[2]/p"
XPATH_GENERAL_COMMENT_REPLY_AGREEMENTS = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/article/article/div[2]/div[2]/div/ul/li[1]/button[2]/span/span[2]"
XPATH_GENERAL_COMMENT_REPLY_ACKNOWLEDGEMENTS = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/article/article/div[2]/div[2]/div/ul/li[2]/button[2]/span/span[2]"
XPATH_GENERAL_COMMENT_REPLY_DISAGREEMENTS = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/article/article/div[2]/div[2]/div/ul/li[3]/button[2]/span/span[2]"
XPATH_GENERAL_COMMENT_REPLY_COUNT = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/article/article/div[2]/div[2]/button[1]/span"
XPATH_GENERAL_COMMENT_REPLY_BUTTON = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/article/article/div[2]/div[2]/button[1]"
XPATH_GENERAL_COMMENT_REPLY_BUTTON_MORE = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/div/div[1]/div/button"

# 返信コメントの要素
XPATH_REPLY_COMMENT_SECTIONS = (
    "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/div/ul/li[i3]"
)
XPATH_REPLY_COMMENT_USERNAME = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/div/ul/li[i3]/div/article/div[2]/div[1]/h2/a"
XPATH_REPLY_COMMENT_POSTED_TIME = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/div/ul/li[i3]/div/article/div[2]/div[1]/time"
XPATH_REPLY_COMMENT_COMMENT_TEXT = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/div/ul/li[i3]/div/article/div[2]/p"
XPATH_REPLY_COMMENT_AGREEMENTS = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/div/ul/li[i3]/div/article/div[2]/div[2]/ul/li[1]/button[2]/span/span[2]"
XPATH_REPLY_COMMENT_ACKNOWLEDGEMENTS = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/div/ul/li[i3]/div/article/div[2]/div[2]/ul/li[2]/button[2]/span/span[2]"
XPATH_REPLY_COMMENT_DISAGREEMENTS = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/div/ul/li[i3]/div/article/div[2]/div[2]/ul/li[3]/button[2]/span/span[2]"
