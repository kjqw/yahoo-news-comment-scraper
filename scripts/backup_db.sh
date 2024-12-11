#!/bin/bash

# バックアップ先ディレクトリを指定
backup_dir="../db_backups"

# 日付を付けたファイル名を作成
backup_file="${backup_dir}/$(date +%Y%m%d_%H%M%S).sql"

# データベース接続情報
db_host="postgresql_db"
db_user="kjqw"
db_name="yahoo_news"

# バックアップ先ディレクトリが存在しない場合は作成
if [ ! -d "$backup_dir" ]; then
    mkdir -p "$backup_dir"
fi

# バックアップ実行
pg_dump -h "$db_host" -U "$db_user" "$db_name" >"$backup_file"

# バックアップ成功の確認
if [ $? -eq 0 ]; then
    echo "バックアップが正常に完了しました: $backup_file"
else
    echo "バックアップに失敗しました"
fi
