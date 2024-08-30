# バグ対処のメモ

## init_driver() が終了しない

`.devcontainer` と同じ階層で`docker restart selenium`を実行したら直った。

```sh
kjqw@DESKTOP-V5HNETV ~/university/B4/study/yahoo_news % docker restart selenium

selenium
kjqw@DESKTOP-V5HNETV ~/university/B4/study/yahoo_news %
```

## 返信を表示するボタンを押した後に表示されるはずの要素が取得できない

`find_all_combinations`関数で、複数のプレースホルダーがあるときにバグっていそう。
解決

## urlが存在しなくなる

2024/8/29夜に<https://news.yahoo.co.jp/articles/ddae7ca185c389a92d2c1136a699a32fe4415094/comments>にアクセスすると、開けていたはずのコメントページが開けなくなった。昼は開けていた。<https://news.yahoo.co.jp/articles/ddae7ca185c389a92d2c1136a699a32fe4415094>のように記事のページも開けなくなってた。このurlはおそらく1週間ほど前に取得したものであるから、記事が削除されたのかもしれない。

## find_elementsとfind_element

ない要素を取得しようとすると、find_elementはエラーを返すが、find_elementsは空のリストを返す。

## 返信が取得できない

8/29には取得できていたものが8/30ではできない。同じコードを実行しているのにできない。返信を表示するボタンが押せていない？
