CREATE TABLE IF NOT EXISTS nodes (
    id SERIAL PRIMARY KEY, -- ID
    node_id INTEGER NOT NULL, -- ノードのID
    k INTEGER NOT NULL, -- 現在の時刻
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
    UNIQUE (node_id, k)
);