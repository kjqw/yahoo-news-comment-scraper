#!/bin/bash

# バックアップファイルのディレクトリを指定
backup_dir="../db_backups"

# データベース接続情報
db_host="postgresql_db"
db_user="kjqw"
default_db_name="yahoo_news"

# バックアップファイルの選択
echo "以下のバックアップファイルから選択してください:"
select backup_file in "$backup_dir"/*.sql; do
    if [ -n "$backup_file" ]; then
        echo "選択されたバックアップファイル: $backup_file"
        break
    else
        echo "有効なファイルを選択してください。"
    fi
done

# 新しいデータベース名の入力を促す
read -p "復元先のデータベース名を入力してください (デフォルト: ${default_db_name}_restore): " new_db_name
new_db_name=${new_db_name:-"${default_db_name}_restore"}

# 確認メッセージ
echo "選択されたバックアップファイル: $backup_file"
echo "復元先データベース名: $new_db_name"
read -p "本当にデータベースを復元しますか？(y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "復元をキャンセルしました"
    exit 0
fi

# 新しいデータベースの作成
echo "新しいデータベースを作成しています: $new_db_name"
createdb -h "$db_host" -U "$db_user" "$new_db_name"

if [ $? -ne 0 ]; then
    echo "データベースの作成に失敗しました: $new_db_name"
    exit 1
fi

# 復元実行
echo "バックアップファイルからデータを復元しています..."
psql -h "$db_host" -U "$db_user" -d "$new_db_name" <"$backup_file"

if [ $? -eq 0 ]; then
    echo "復元が正常に完了しました: $new_db_name"
else
    echo "復元に失敗しました"
    exit 1
fi
