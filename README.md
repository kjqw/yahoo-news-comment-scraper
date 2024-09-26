# Yahoo!ニュースコメントスクレイピングツール

このリポジトリは、Yahoo!ニュースのコメントをスクレイピングし、そのデータを整形・保存するためのツールを開発することを目的としています。

---

## 現在の機能

- Yahoo!ニュースの記事のコメントページから以下の情報をスクレイピングする機能があります。
  - コメント内容
  - 投稿者
  - 投稿日時
  - 反応（「なるほど」「共感した」「うーん」の数）
  - 返信数
  - 返信コメント
  - 返信コメントに対する反応（「なるほど」「共感した」「うーん」の数）

## 環境構築

- `git clone https://github.com/kjqw/yahoo-news-comment-scraper.git`などしてリポジトリをクローン
- `code yahoo-news-comment-scraper`などしてVSCodeでプロジェクトを開く
- 右下に表示される「Reopen in Container」をクリックしてDockerコンテナを起動
- スクレイピングに使うライブラリとともに、私の開発環境が入ったDockerコンテナが起動します
- これでスクレイピングができる環境が整いました

---

## 実行手順

### article_comment_scraper.py

開発中のスクリプトである。

`scraping/article_comment_scraper.py`を実行することで、特定のYahoo!ニュースのコメントをスクレイピングし、データを`pkl`形式で保存することができます。以下を引数に指定することで、スクレイピング対象の記事を指定できます。指定しないとデフォルト値が適用されます。

- `--url`: スクレイピング対象の記事のコメントページのURL
- `--max_comments`: 取得するコメントの最大値
- `--max_replies`: 取得する返信コメントの最大値
- `--order`: コメントの並び順（`recommended`または`newer`）
- `--timeout`: ページ読み込みのタイムアウト時間（秒）
- `--save_path`: スクレイピング結果を保存するディレクトリのパス

例えば、以下のように実行することで、指定したURLの記事のコメントをスクレイピングし、`data/`ディレクトリに`comments.pkl`として保存することができます。

```sh
python scraping/article_comment_scraper.py --url https://news.yahoo.co.jp/articles/d15ad8cacf5255134e2890075ca636835cfdfa23/comments --max_comments 50 --max_replies 10 --save_path data/comments_argparse.pkl
```

**注意**

- 記事は消えていたりすることもあるので、URLは最新のものを指定してください。
- pkl形式のファイルを読み込むときは、`scraping/classes.py`に定義されているクラスをインポートする必要があります。
- 強制終了などによって`driver.quit()`が実行されない場合、次回のスクレイピング時に固まることがあります。その場合は、`docker restart selenium`を実行してください。

---

### 主要ファイル

- `scraping/article_link_scraper.py`: Yahoo!ニュースのランキングページから記事のリンクをスクレイピングするスクリプト
- `scraping/article_scraper.py`: 記事のリンクから記事の内容をスクレイピングするスクリプト
- `scraping/article_comment_scraper.py`: 記事のコメントページからコメントをスクレイピングするスクリプト
- `scraping/user_comment_scraper.py`: ユーザーページからユーザーのコメントをスクレイピングするスクリプト
- `scraping/xpaths/`: スクレイピングに使用するXPathの定義ファイル
- `scraping/functions.py`: スクレイピングに使用する関数の定義ファイル
- `scraping/classes.py`: スクレイピングに使用するクラスの定義ファイル
- `memo/`: 開発中のメモやバグ情報を記載したもの

---

## 今後の予定

- データの整形・分析機能の追加
- スクレイピング対象の多様化
  - ユーザーごとのコメント取得
  - 記事自体のスクレイピング
