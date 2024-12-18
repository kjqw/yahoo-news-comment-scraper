-- ユーザーの遷移モデルの学習データ用のテーブルを作成する

CREATE TABLE IF NOT EXISTS articles_llm_output (
    article_id INTEGER REFERENCES articles (article_id),
    article_title TEXT,
    -- article_title_vector FLOAT[],
    article_content TEXT,
    article_content_vector FLOAT[],
    posted_time TEXT,
    normalized_posted_time TIMESTAMP,
    UNIQUE (article_id)
);

CREATE TABLE IF NOT EXISTS comments_llm_output (
    comment_id INTEGER REFERENCES comments (comment_id),
    article_id INTEGER REFERENCES articles_llm_output (article_id),
    parent_comment_id INTEGER REFERENCES comments_llm_output (comment_id),
    user_id INTEGER REFERENCES users (user_id),
    posted_time TEXT,
    normalized_posted_time TIMESTAMP,
    is_time_uncertain BOOLEAN,
    comment_content TEXT,
    comment_content_vector FLOAT[],
    UNIQUE (
        comment_id,
        article_id,
        user_id,
        comment_content
    )
);

CREATE TABLE IF NOT EXISTS training_data (
    user_id INTEGER REFERENCES users (user_id),
    article_id INTEGER REFERENCES articles_llm_output (article_id),
    article_content_vector FLOAT[] REFERENCES articles_llm_output (article_content_vector),
    parent_comment_id INTEGER REFERENCES comments_llm_output (comment_id),
    parent_comment_content_vector FLOAT[] REFERENCES comments_llm_output (comment_content_vector),
    comment_id INTEGER REFERENCES comments_llm_output (comment_id),
    comment_content_vector FLOAT[] REFERENCES comments_llm_output (comment_content_vector),
    normalized_posted_time TIMESTAMP REFERENCES comments_llm_output (normalized_posted_time)
);