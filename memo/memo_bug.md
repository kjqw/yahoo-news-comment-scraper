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
