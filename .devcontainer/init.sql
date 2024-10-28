CREATE TABLE IF NOT EXISTS articles (
    article_id SERIAL PRIMARY KEY,
    link TEXT UNIQUE NOT NULL,
    genre TEXT,
    title TEXT,
    author TEXT,
    author_link TEXT,
    posted_time TEXT,
    updated_time TEXT,
    ranking INTEGER,
    content TEXT,
    comments_count INTEGER,
    comments_count_per_hour INTEGER,
    learn_count INTEGER,
    clarity_count INTEGER,
    new_perspective_count INTEGER,
    scraped_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS comments (
    comment_id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles (article_id) ON DELETE CASCADE,
    parent_comment_id INTEGER,
    username TEXT,
    user_link TEXT,
    posted_time TEXT,
    content TEXT,
    agreements_count INTEGER,
    acknowledgements_count INTEGER,
    disagreements_count INTEGER,
    reply_count INTEGER,
    scraped_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS related_articles (
    article_id INTEGER REFERENCES articles (article_id) ON DELETE CASCADE,
    related_article_id INTEGER REFERENCES articles (article_id) ON DELETE CASCADE,
    PRIMARY KEY (
        article_id,
        related_article_id
    )
);

CREATE TABLE IF NOT EXISTS read_also_articles (
    article_id INTEGER REFERENCES articles (article_id) ON DELETE CASCADE,
    read_also_article_id INTEGER REFERENCES articles (article_id) ON DELETE CASCADE,
    PRIMARY KEY (
        article_id,
        read_also_article_id
    )
);

CREATE TABLE IF NOT EXISTS replies (
    comment_id INTEGER REFERENCES comments (comment_id) ON DELETE CASCADE,
    reply_comment_id INTEGER REFERENCES comments (comment_id) ON DELETE CASCADE,
    PRIMARY KEY (comment_id, reply_comment_id)
);