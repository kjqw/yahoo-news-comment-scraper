CREATE TABLE IF NOT EXISTS metadata (
    metadata_id SERIAL PRIMARY KEY, -- ID
    article_num INTEGER, -- 記事数
    user_num INTEGER, -- ユーザー数
    state_dim INTEGER, -- 状態ベクトルの次元
    k_max INTEGER, -- 最大時刻
    identifier INTEGER, -- 識別子
    add_noise BOOLEAN, -- ノイズを加えるかどうか
    is_discrete BOOLEAN, -- 状態ベクトルが離散値かどうか
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- タイムスタンプ
    UNIQUE (
        article_num,
        user_num,
        state_dim,
        k_max,
        identifier,
        add_noise,
        is_discrete
    )
);

CREATE TABLE IF NOT EXISTS nodes (
    id SERIAL PRIMARY KEY, -- ID
    node_id INTEGER NOT NULL, -- ノードのID
    k INTEGER NOT NULL, -- 現在の時刻
    metadata_id INTEGER NOT NULL REFERENCES metadata (metadata_id), -- メタデータID
    node_type VARCHAR(20) NOT NULL, -- ノードの種類（'article' または 'comment'）
    parent_ids INTEGER[], -- 親ノードのID
    parent_ks INTEGER[], -- 親ノードの時刻
    state_dim INTEGER, -- 状態ベクトルの次元
    k_max INTEGER, -- 最大時刻
    state FLOAT[], -- 状態ベクトル
    strength FLOAT, -- 影響度
    W_p FLOAT[] [], -- 重み行列
    W_q FLOAT[] [], -- 重み行列
    W_s FLOAT[] [], -- 重み行列
    b FLOAT[], -- バイアスベクトル
    UNIQUE (node_id, k, metadata_id)
);

CREATE TABLE IF NOT EXISTS params (
    node_id INTEGER NOT NULL, -- ユーザーID
    metadata_id INTEGER NOT NULL, -- メタデータID
    w_p_true FLOAT[] [], -- 真の重み行列
    w_q_true FLOAT[] [], -- 真の重み行列
    w_s_true FLOAT[] [], -- 真の重み行列
    b_true FLOAT[], -- 真のバイアスベクトル
    w_p_est FLOAT[] [], -- 推定された重み行列
    w_q_est FLOAT[] [], -- 推定された重み行列
    w_s_est FLOAT[] [], -- 推定された重み行列
    b_est FLOAT[] -- 推定されたバイアスベクトル
);