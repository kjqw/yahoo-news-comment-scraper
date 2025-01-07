# ユーザーの情報のXPath
XPATH_USER_NAME = [
    "/html/body/div[1]/div/main/div[1]/div/div[1]/div/div[2]/p[1]",
    "/html/body/div[1]/div/main/div[1]/div/div[2]/div/div[2]/p[1]",
]
XPATH_TOTAL_COMMENT_COUNT = [
    "/html/body/div[1]/div/main/div[1]/div/div[2]/div/span[2]",
    "/html/body/div[1]/div/main/div[1]/div/div[3]/div/span[2]",
]

XPATH_TOTAL_AGREEMENTS = [
    "/html/body/div[1]/div/main/div[1]/div/div[2]/ul/li[1]/span/span[2]",
    "/html/body/div[1]/div/main/div[1]/div/div[3]/ul/li[1]/span/span[2]",
]
XPATH_TOTAL_ACKNOWLEDGEMENTS = [
    "/html/body/div[1]/div/main/div[1]/div/div[2]/ul/li[2]/span/span[2]",
    "/html/body/div[1]/div/main/div[1]/div/div[3]/ul/li[2]/span/span[2]",
]
XPATH_TOTAL_DISAGREEMENTS = [
    "/html/body/div[1]/div/main/div[1]/div/div[2]/ul/li[3]/span/span[2]",
    "/html/body/div[1]/div/main/div[1]/div/div[3]/ul/li[3]/span/span[2]",
]

# コメント部分のXPath
XPATH_COMMENT_SECTIONS = "/html/body/div[1]/div/main/div[1]/div/ul/li"

# ユーザーのコメントが、記事に対するコメントなのかコメントに対する返信コメントなのかによって、articleタグの数が変わる
RELATIVE_XPATH_COMMENT_SECTION = "article"
# "/article[n]/div/div/div[2]/div/a/span[1]"の値が"返信"の場合は、それは記事に対するコメントであり、値が存在しない場合は返信のコメントであると判断できそう。nは2以上。n=1は記事の情報

# 以下は RELATIVE_XPATH_COMMENT_SECTION からの相対パスである
RELATIVE_XPATH_ARTICLE_LINK = "a"
RELATIVE_XPATH_ARTICLE_TITLE = "a/div/h2/p[1]"
RELATIVE_XPATH_ARTICLE_AUTHOR = "a/div/h2/p[2]/span"
RELATIVE_XPATH_ARTICLE_POSTED_TIME = "a/div/h2/p[2]/time"
RELATIVE_XPATH_ARTICLE_COMMENT_PAGE_LINK = "div/a"

RELATIVE_XPATH_COMMENT_REPLY_COMMENT_LINK = "div/div[1]/a"  # ないこともある
# 返信先のコメントが削除されている場合は "div/div[1]/p" に "削除されたコメントです" という文字列が入っている
RELATIVE_XPATH_COMMENT_REPLY_COMMENT_TEXT = "div/div[1]/p"

RELATIVE_XPATH_COMMENT_POSTED_TIME_1 = "div/div/div[1]/a"  # 記事に対するコメントの場合
RELATIVE_XPATH_COMMENT_POSTED_TIME_2 = (
    "div/div/div[1]/time"  # コメントに対するコメントの場合
)

RELATIVE_XPATH_COMMENT_TEXT = (
    "div/div/div[2]/p"  # 記事に対するコメントでも返信コメントでもパスは同じ
)
RELATIVE_XPATH_COMMENT_AGREEMENTS_1 = "div/div/div[2]/div/ul/li[1]/div/span/span[2]"  # 記事に対するコメントの「共感した」の数
RELATIVE_XPATH_COMMENT_AGREEMENTS_2 = "div/div/div[2]/ul/li[1]/div/span/span[2]"  # コメントに対するコメントの「共感した」の数
RELATIVE_XPATH_COMMENT_ACKNOWLEDGEMENTS_1 = "div/div/div[2]/div/ul/li[2]/div/span/span[2]"  # 記事に対するコメントの「なるほど」の数
RELATIVE_XPATH_COMMENT_ACKNOWLEDGEMENTS_2 = "div/div/div[2]/ul/li[2]/div/span/span[2]"  # コメントに対するコメントの「なるほど」の数
RELATIVE_XPATH_COMMENT_DISAGREEMENTS_1 = "div/div/div[2]/div/ul/li[3]/div/span/span[2]"  # 記事に対するコメントの「反対した」の数
RELATIVE_XPATH_COMMENT_DISAGREEMENTS_2 = "div/div/div[2]/ul/li[3]/div/span/span[2]"  # コメントに対するコメントの「反対した」の数
RELATIVE_XPATH_COMMENT_REPLY_COUNT = "div/div/div[2]/div/a/span[2]"  # これの有無で記事に対するコメントか返信コメントかが判断できる

XPATH_COMMENT_MORE_BUTTON = (
    "/html/body/div[1]/div/main/div[1]/div/div[4]/div/span/button"
)

# ------------------------------------------------------------
# 返信先のコメントの独立ページのXPath
XPATH_REPLY_COMMENT_SECTION = "/html/body/div[1]/div/main/div[1]/div/div[1]"

RELATIVE_XPATH_REPLY_COMMENT_USERNAME = "article/div/div[2]/div/a"
RELATIVE_XPATH_REPLY_COMMENT_POSTED_TIME = "article/div/div[2]/div/time"
RELATIVE_XPATH_REPLY_COMMENT_TEXT = "article/p"
RELATIVE_XPATH_REPLY_COMMENT_AGREEMENTS = "div[2]/ul/li[1]/button[2]/span/span[2]"
RELATIVE_XPATH_REPLY_COMMENT_ACKNOWLEDGEMENTS = "div[2]/ul/li[2]/button[2]/span/span[2]"
RELATIVE_XPATH_REPLY_COMMENT_DISAGREEMENTS = "div[2]/ul/li[3]/button[2]/span/span[2]"
