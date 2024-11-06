DROP TABLE IF EXISTS articles,
biases,
comments,
node_relations,
node_states,
weights CASCADE;

DROP SEQUENCE IF EXISTS articles_article_id_seq,
biases_id_seq,
comments_comment_id_seq,
node_relations_id_seq,
node_states_id_seq,
weights_id_seq CASCADE;