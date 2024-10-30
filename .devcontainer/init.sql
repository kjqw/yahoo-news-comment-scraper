-- バージョン管理用テーブル
CREATE TABLE IF NOT EXISTS versions (
    version_id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- スクレイピング結果の保存用テーブル
CREATE TABLE IF NOT EXISTS articles (
    article_id SERIAL PRIMARY KEY,
    link TEXT NOT NULL,
    title TEXT,
    author TEXT,
    posted_time TEXT,
    ranking INTEGER,
    comments_count_per_hour INTEGER,
    scraped_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version_id INTEGER REFERENCES versions (version_id) ON DELETE CASCADE -- バージョン参照
);

-- エラーログ用テーブル
CREATE TABLE IF NOT EXISTS errors (
    error_id SERIAL PRIMARY KEY,
    error_message TEXT,
    error_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    function_name TEXT,
    article_id INTEGER REFERENCES articles (article_id) ON DELETE SET NULL
);