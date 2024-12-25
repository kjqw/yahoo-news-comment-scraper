CREATE TABLE IF NOT EXISTS training_data_raw (
    user_id INTEGER,
    article_id INTEGER,
    article_content TEXT,
    parent_comment_id INTEGER,
    parent_comment_content TEXT,
    comment_id INTEGER,
    comment_content TEXT,
    normalized_posted_time TIMESTAMP,
    UNIQUE (comment_id)
);

CREATE TABLE IF NOT EXISTS training_data_vectorized (
    user_id INTEGER,
    article_id INTEGER,
    article_content_vector FLOAT[],
    parent_comment_id INTEGER,
    parent_comment_content_vector FLOAT[],
    comment_id INTEGER,
    comment_content_vector FLOAT[],
    normalized_posted_time TIMESTAMP,
    UNIQUE (comment_id)
);