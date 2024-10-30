# メモ

以下がないと`psql`コマンドが使えない。

```sh
sudo apt install postgresql-client
```

起動したコンテナで以下を実行し、パスワードを入力するとデータベースに接続できる。パスワードは`docker-compose.yml`で指定したもの。

```sh
psql -h postgresql_db -U kjqw -d yahoo_news
```

ホスト環境で空の`yahoo-news-comment-scraper/db_data`を作成し、コンテナを起動するとsql関連のファイルがそこに保存される。VSCodeで`db_data`を開いても中身が見えないが、rootユーザーで`ls`コマンドなどを実行すれば中身が見える。

フォルダを作成したときは所有者は`kjqw`であったが、コンテナを起動すると所有者が`999`に変わった。

```sh
drwx------ 19  999 kjqw 4.0K Oct 28 22:32 db_data
```
