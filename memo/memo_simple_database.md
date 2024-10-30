# メモ

## ブランチの説明

いきなり全機能がある実装は難しいので、記事のリンクを保存するだけのシンプルなデータベースを作成する。
`feature/database`あたりのコミットがぐちゃぐちゃだが、気にしないことにする。これから気を付ける。
ブランチを切るたびにそのブランチの説明を記した`memo*.md`をコミットをすると、開発の記録が分かりやすく残りそう。

## バグ

`init.sql`を変更してコンテナを再起動するだけでは変更が反映されないので、データベースのデータを削除してからコンテナを再起動する。

```sh
sudo rm -r /workspace/yahoo-news-comment-scraper/db_data && mkdir /workspace/yahoo-news-comment-scraper/db_data
```

`db_data/`を消して作り直すだけではなぜかうまくいかなかった。`docker-compose.yaml`があるディレクトリで`docker compose down -v`を実行すると、変更が反映された。

## データベースに接続するコマンド

```sh
psql -h postgresql_db -U kjqw -d yahoo_news
```
