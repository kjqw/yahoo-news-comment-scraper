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
8/31にはできるようになっていた。原因は不明。
要素が表示されるまで待機すれば安定して取得できるようになった。

## 返信数が取得できない

reply_countが空で返ってくる。article_comment_scraper.pyでは常に空で返ってくるが、ほぼ同じはずの以下のようなコードでは取得できる。

```python
try:
    driver = utils.init_driver()
    driver.get(url)

    comment_sections = utils.get_comment_sections(driver)
    reply_counts = []
    reply_counts_text = []
    for comment_section in comment_sections:
        reply_count_element = comment_section.find_element(
            By.XPATH, RELATIVE_XPATH_GENERAL_COMMENT_REPLY_COUNT
        )
        reply_counts.append(reply_count_element)
        reply_counts_text.append(reply_count_element.text)

finally:
    driver.quit()
```

`get_reply_comment_sections`の前に返信数を取得すると取得できた。`get_reply_comment_sections`の中で返信ボタンをクリックする動作があるが、それによってwebelementに何らかの変化が起きているのかもしれない。

## 記事が複数ページあると「学びがある」数などが取得できない

ページ遷移のボタンの有無でdivがずれる

複数のxpathを試す処理で一応解決

## デバイスによってGPUが使えない

自宅デスクトップPC、研究室ノートPC1ではGPUが使えるが、研究室ノートPC2では使えなかった。研究室ノートPC2でコンテナをrebuildしたら研究室ノートPC2でGPUが使えるようになった。ほかのデバイスで使えるかは未確認。

## 削除された記事への対応

応急処置として、削除される記事はネガティブであろうという仮定のもと、ネガティブな記事として扱う。

## training_data_rawテーブルのarticle_contentがNoneなのにtraining_data_vectorizedテーブルにデータが入っている

Noneという文字列をLLMが分類しているぽい。データベース格納時にNullとしないといけない。

## 保存しようとしている文章にシングルクォーテーションが含まれていると保存できない

<!-- TODO -->
`execute_query()`の引数を`query`と`params`にわければ解決するかもしれないが、`execute_query()`が使われている箇所が多いので変更が面倒である。

```sh
  9%|▉         | 263/2939 [13:16<2:24:16,  3.23s/it]
データベースエラー: syntax error at or near "S"
LINE 13: ...の戦績を残している。直近では2024年6月に『KNUCKLE'S 16』のキ...
```

## バッチ学習

親コメントが存在しないときに parent_comment_content_vector を [2, 2, 2] に設定し、[2, 2, 2] が入力されたら違う処理をするようにしていたが、バッチ化によってこの処理はされていなさそう。入力されるデータは $n\times 3$ の行列であるから、[2, 2, 2] が入力されることはなさそう。なので、親コメントがないときは [0, 1, 0] を入力して試してみた。コメントを読まないことと中立のコメントを読むことは区別したかったが、20250122151749 の学習ではそうした。

そうしても行列のモデルでは教師データの平均値に収束するような結果になった。
