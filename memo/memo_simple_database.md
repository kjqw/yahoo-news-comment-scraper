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

`db_data/`を消して作り直すだけではなぜかうまくいかなかった。`docker-compose.yaml`があるディレクトリで以下を実行すると、変更が反映された。

```sh
docker compose down -v
```

## データベースに接続するコマンド

```sh
psql -h postgresql_db -U kjqw -d yahoo_news
```

## 取得するコメントについて

自分のコメントに対する返信は取得しないことにする。

- 理由
  - 取得しようとすると、投稿時間や「共感した」数のxpathが煩雑で、条件分岐が複雑になる
  - 自分のコメントに対する返信は、厳密には自分のコメントについた他人のコメントに対する返信であるが、どの他人のコメントに対する返信であるかがシステム上分かりにくい
    - 自分のコメントに対する返信に関してスクレイピングで取得できるのは、自分のコメント・自分の返信コメント、のみであり、自分の返信コメントが何に対しての返信であるかは簡単には分からない
    - コメント内容に返信先が書かれていることもあるが、全ての場合でそうではない

## スクレイピングの手順

- `scraping/article_link_scraper.py`でコメントランキング上位の記事のリンクを取得
  - 取得された記事について、`scraping/article_scraper.py`で記事の内容を取得
  - 取得された記事について、`scraping/article_comment_scraper.py`でコメントを取得
    - 取得されたコメントのユーザーについて、`scraping/user_comment_scraper.py`でそのユーザーが過去にしたコメントを取得

## normalized_posted_timeが同じ場合

スクレイピングはコメントの新しい順にやっているので、`normalized_posted_time`が同じ場合は以下のようにして並び替えると古い順になる。

```sql
ORDER BY "normalized_posted_time" ASC, comment_id DESC
```
