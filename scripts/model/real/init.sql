-- ユーザーの遷移モデルの学習データ用のテーブルを作成する

CREATE TABLE IF NOT EXISTS articles_llm_output (
    article_id INTEGER,
    article_title TEXT,
    -- article_title_vector FLOAT[],
    article_content TEXT,
    article_content_vector FLOAT[],
    posted_time TEXT,
    normalized_posted_time TIMESTAMP,
    UNIQUE (article_id)
);

CREATE TABLE IF NOT EXISTS comments_llm_output (
    comment_id INTEGER,
    article_id INTEGER,
    parent_comment_id INTEGER,
    user_id INTEGER,
    posted_time TEXT,
    normalized_posted_time TIMESTAMP,
    is_time_uncertain BOOLEAN,
    comment_content TEXT,
    comment_content_vector FLOAT[],
    UNIQUE (comment_id)
);

CREATE TABLE IF NOT EXISTS training_data (
    user_id INTEGER,
    article_id INTEGER,
    article_content_vector FLOAT[],
    parent_comment_id INTEGER,
    parent_comment_content_vector FLOAT[],
    comment_id INTEGER,
    comment_content_vector FLOAT[],
    normalized_posted_time TIMESTAMP,
    UNIQUE (user_id, article_id, parent_comment_id, comment_id)
);