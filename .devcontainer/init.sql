CREATE TABLE IF NOT EXISTS articles (
    article_id SERIAL PRIMARY KEY,
    article_link TEXT,
    article_title TEXT,
    author TEXT,
    posted_time TEXT,
    ranking INTEGER,
    comment_count_per_hour INTEGER,
    total_comment_count_with_reply INTEGER,
    total_comment_count_without_reply INTEGER,
    scraped_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS comments (
    comment_id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles (article_id) ON DELETE CASCADE,
    parent_comment_id INTEGER REFERENCES comments (comment_id) ON DELETE CASCADE,
    username TEXT,
    user_link TEXT,
    posted_time TEXT,
    comment_text TEXT,
    agreements_count INTEGER,
    acknowledgements_count INTEGER,
    disagreements_count INTEGER,
    reply_count INTEGER,
    scraped_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);