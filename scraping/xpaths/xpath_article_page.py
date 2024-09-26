# ジャンルのXPath
# 選択されているジャンルの要素は、liのclass属性が他のものと違う
XPATH_GENRE = "/html/body/div[1]/header/nav/div[2]/ul[1]/li/a"

# 記事部分の要素
ID_ARTICLE_SECTION = "contentsWrap"
XPATH_ARTICLE_SECTION = "/html/body/div[1]/div/main/div[1]/div"
RELATIVE_XPATH_ARTICLE_TITLE = "article/header/h1"
RELATIVE_XPATH_ARTICLE_AUTHOR = "article/footer/a"
RELATIVE_XPATH_ARTICLE_POSTED_TIME = "article/header/div/div/p/time"
RELATIVE_XPATH_ARTICLE_UPDATED_TIME = "article/footer/div/time"

RELATIVE_XPATH_ARTICLE_COMMENT_COUNT = "article/header/div/div/div[1]/button[1]/span"
ID_ARTICLE_CONTENT = "highLightSearchTarget"
RELATIVE_XPATH_ARTICLE_CONTENT = "article/div[1]/div/p"
RELATIVE_XPATH_ARTICLE_PAGE_COUNT = (
    "article/div[3]/div/p/span"  # ページ数が複数ある場合のみ存在
)
RELATIVE_XPATH_ARTICLE_LERAN_COUNT = "/article/div[3]/ul/li[1]/div/span"
RELATIVE_XPATH_ARTICLE_CLARITY_COUNT = "article/div[3]/ul/li[2]/div/span"
RELATIVE_XPATH_ARTICLE_NEW_PERSPECTIVE_COUNT = "article/div[3]/ul/li[3]/div/span"

RELATIVE_XPATH_RELATED_ARTICLES = "article/section/ul/li/a"
RELATIVE_XPATH_READ_ALSO_ARTICLE = "aside/div/ul/li/a"
RELATIVE_XPATH_READ_ALSO_ARTICLE_COMMENT_COUNT = (
    "aside/div/ul/li/a/div[2]/div[2]/div/span[1]"  # ないこともある
)
RELATIVE_XPATH_READ_ALSO_ARTICLE_AUTHOR = "aside/div/ul/li/a/div[2]/div[2]/div/span[2]"  # コメント数の要素がない場合はspan[2]ではなくspan[1]になる
RELATIVE_XPATH_READ_ALSO_ARTICLE_POSTED_TIME = (
    "aside/div/ul/li/a/div[2]/div[2]/div/time"
)
XPATH_READ_ALSO_ARTICLE_MORE_BUTTON = "aside/div/div[2]/button"
