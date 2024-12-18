CREATE TABLE IF NOT EXISTS articles (
    article_id SERIAL PRIMARY KEY,
    article_link TEXT,
    article_title TEXT,
    author TEXT,
    posted_time TEXT,
    updated_time TEXT,
    ranking INTEGER,
    article_genre TEXT,
    article_content TEXT,
    comment_count_per_hour INTEGER,
    total_comment_count_with_reply INTEGER,
    total_comment_count_without_reply INTEGER,
    learn_count INTEGER,
    clarity_count INTEGER,
    new_perspective_count INTEGER,
    -- scraping_status BOOLEAN DEFAULT FALSE, -- スクレイピング成功フラグ
    scraped_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (article_link)
);

CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    username TEXT,
    user_link TEXT,
    total_comment_count INTEGER,
    total_agreements_count INTEGER,
    total_acknowledgements_count INTEGER,
    total_disagreements_count INTEGER,
    scraped_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_link)
);

CREATE TABLE IF NOT EXISTS comments (
    comment_id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles (article_id),
    parent_comment_id INTEGER REFERENCES comments (comment_id),
    user_id INTEGER REFERENCES users (user_id),
    username TEXT,
    user_link TEXT,
    posted_time TEXT,
    normalized_posted_time TIMESTAMP,
    is_time_uncertain BOOLEAN,
    comment_content TEXT,
    agreements_count INTEGER,
    acknowledgements_count INTEGER,
    disagreements_count INTEGER,
    reply_count INTEGER,
    -- scraping_status BOOLEAN DEFAULT FALSE, -- スクレイピング成功フラグ
    scraped_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (
        article_id,
        user_link,
        comment_content
    )
);