-- 記事テーブル
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

-- コメントノード（ユーザーコメント）テーブル
CREATE TABLE IF NOT EXISTS comments (
    comment_id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles (article_id) ON DELETE CASCADE, -- 関連する記事ID
    parent_comment_id INTEGER REFERENCES comments (comment_id) ON DELETE CASCADE, -- 親コメントID
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

-- ノード（記事およびコメント）の状態管理用テーブル
CREATE TABLE IF NOT EXISTS node_states (
    id SERIAL PRIMARY KEY,
    node_id INTEGER NOT NULL, -- ノードのID（articlesまたはcommentsのID）
    node_type VARCHAR(20) NOT NULL, -- ノードの種類（'article' または 'comment'）
    time_step INT, -- 時刻 k
    state FLOAT[], -- 状態ベクトル
    strength FLOAT, -- 影響度
    noise FLOAT[], -- ノイズ
    UNIQUE (node_id, node_type, time_step) -- 各ノードの時刻ごとの状態
);

-- 親子関係を管理するテーブル
CREATE TABLE IF NOT EXISTS node_relations (
    id SERIAL PRIMARY KEY,
    parent_node_id INTEGER, -- 親ノードのID
    parent_node_type VARCHAR(20) NOT NULL, -- 親ノードの種類（'article' または 'comment'）
    child_node_id INTEGER, -- 子ノードのID
    child_node_type VARCHAR(20), -- 子ノードの種類
    time_step INT NOT NULL, -- 時刻 k
    UNIQUE (
        parent_node_id,
        parent_node_type,
        child_node_id,
        child_node_type,
        time_step
    )
);

-- コメントノードの重み行列を管理するテーブル
CREATE TABLE IF NOT EXISTS weights (
    id SERIAL PRIMARY KEY,
    node_id INTEGER NOT NULL REFERENCES comments (comment_id) ON DELETE CASCADE, -- コメントノードID
    weight_type VARCHAR(10) NOT NULL, -- 重みの種類（'W_p', 'W_q', 'W_s'）
    weight_matrix FLOAT[][] NOT NULL, -- 重み行列
    UNIQUE (node_id, weight_type)
);

-- コメントノードのバイアスベクトルを管理するテーブル
CREATE TABLE IF NOT EXISTS biases (
    id SERIAL PRIMARY KEY,
    node_id INTEGER NOT NULL REFERENCES comments (comment_id) ON DELETE CASCADE, -- コメントノードID
    bias_vector FLOAT[] NOT NULL, -- バイアスベクトル
    UNIQUE (node_id)
);