# 記事ページの要素
ARTICLE_TITLE = "/html/body/div[1]/div/main/div[1]/div[1]/article/header/h1/a"
ARTICLE_AUTHOR = "/html/body/div[1]/div/main/div[1]/div[1]/article/header/p/a"
ARTICLE_POSTED_TIME = (
    "/html/body/div[1]/div/main/div[1]/div[1]/article/header/p/span/time"
)

# コメント数
PAGE_COMMENT_COUNT_1 = "/html/body/div[1]/div/main/div[1]/div[1]/article/div[6]/div/p"
PAGE_COMMENT_COUNT_OTHERS = (
    "/html/body/div[1]/div/main/div[1]/div[1]/article/div[4]/div/p"
)
TOTAL_COMMENT_COUNT_1 = "/html/body/div[1]/div/main/div[1]/div[1]/article/div[4]/div/p"
TOTAL_COMMENT_COUNT_OTHERS = (
    "/html/body/div[1]/div/main/div[1]/div[1]/article/div[2]/div/p"
)

# 専門家コメントの要素
EXPERT_USERNAME = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i1]/article/div[2]/div[1]/h2/a"
EXPERT_POSTED_TIME = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i1]/article/div[2]/div[1]/time/a"
EXPERT_TYPE = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i1]/article/div[2]/div[1]/p"
EXPERT_COMMENT_BUTTON = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i1]/article/div[2]/p/button"
EXPERT_COMMENT_TEXT = (
    "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i1]/article/div[2]/p"
)
EXPERT_AGREEMENTS = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i1]/article/div[2]/div[2]/button/span"

# 一般コメントの要素
GENERAL_COMMENT_SECTIONS = (
    "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]"
)
GENERAL_COMMENT_USERNAME = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/article/article/div[2]/div[1]/h2/a"
GENERAL_COMMENT_POSTED_TIME = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/article/article/div[2]/div[1]/time/a"
GENERAL_COMMENT_COMMENT_TEXT = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/article/article/div[2]/p"
GENERAL_COMMENT_REPLY_AGREEMENTS = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/article/article/div[2]/div[2]/div/ul/li[1]/button[2]/span/span[2]"
GENERAL_COMMENT_REPLY_ACKNOWLEDGEMENTS = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/article/article/div[2]/div[2]/div/ul/li[2]/button[2]/span/span[2]"
GENERAL_COMMENT_REPLY_DISAGREEMENTS = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/article/article/div[2]/div[2]/div/ul/li[3]/button[2]/span/span[2]"
GENERAL_COMMENT_REPLY_COUNT = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/article/article/div[2]/div[2]/button[1]/span"
GENERAL_COMMENT_REPLY_BUTTON = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/article/article/div[2]/div[2]/button[1]"
GENERAL_COMMENT_REPLY_BUTTON_MORE = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/div/div[1]/div/button"

# 返信コメントの要素
REPLY_COMMENT_SECTIONS = (
    "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/div/ul/li[i3]"
)
REPLY_COMMENT_USERNAME = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/div/ul/li[i3]/div/article/div[2]/div[1]/h2/a"
REPLY_COMMENT_POSTED_TIME = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/div/ul/li[i3]/div/article/div[2]/div[1]/time"
REPLY_COMMENT_COMMENT_TEXT = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/div/ul/li[i3]/div/article/div[2]/p"
REPLY_COMMENT_AGREEMENTS = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/div/ul/li[i3]/div/article/div[2]/div[2]/ul/li[1]/button[2]/span/span[2]"
REPLY_COMMENT_ACKNOWLEDGEMENTS = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/div/ul/li[i3]/div/article/div[2]/div[2]/ul/li[2]/button[2]/span/span[2]"
REPLY_COMMENT_DISAGREEMENTS = "/html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i2]/div/ul/li[i3]/div/article/div[2]/div[2]/ul/li[3]/button[2]/span/span[2]"
