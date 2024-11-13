CREATE TABLE IF NOT EXISTS articles (
    article_id SERIAL PRIMARY KEY,
    article_link TEXT,
    version INTEGER DEFAULT 1, -- バージョン番号
    scraped_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (
        article_link,
        version
    )
);

CREATE TABLE IF NOT EXISTS articles_version_history (
    article_id INTEGER REFERENCES articles (article_id),
    version INTEGER NOT NULL, -- バージョン番号
    article_link TEXT,
    article_title TEXT,
    author TEXT,
    posted_time TEXT,
    ranking INTEGER,
    article_genre TEXT,
    article_content TEXT,
    comment_count_per_hour INTEGER,
    total_comment_count_with_reply INTEGER,
    total_comment_count_without_reply INTEGER,
    scraping_status BOOLEAN DEFAULT FALSE, -- スクレイピング成功フラグ
    scraped_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (article_id, version) -- 複合主キー
);

CREATE TABLE IF NOT EXISTS comments (
    comment_id SERIAL PRIMARY KEY,
    article_link TEXT REFERENCES articles (article_link),
    parent_comment_id INTEGER REFERENCES comments (comment_id),
    comment_content TEXT,
    version INTEGER DEFAULT 1, -- バージョン番号
    scraped_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (
        article_link,
        comment_content,
        version
    )
);

CREATE TABLE IF NOT EXISTS comments_version_history (
    comment_id INTEGER REFERENCES comments (comment_id),
    article_link TEXT REFERENCES articles (article_link),
    parent_comment_id INTEGER REFERENCES comments_version_history (comment_id),
    version INTEGER NOT NULL, -- バージョン番号
    username TEXT,
    user_link TEXT,
    posted_time TEXT,
    comment_content TEXT,
    agreements_count INTEGER,
    acknowledgements_count INTEGER,
    disagreements_count INTEGER,
    reply_count INTEGER,
    scraping_status BOOLEAN DEFAULT FALSE, -- スクレイピング成功フラグ
    scraped_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (comment_id, version) -- 複合主キー
);