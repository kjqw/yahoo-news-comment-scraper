#!/bin/bash

# バックアップファイルのディレクトリを指定
backup_dir="../db_backups"

# デフォルトのデータベース情報
default_db_host="postgresql_db"
default_db_user="kjqw"
default_db_name="yahoo_news"

# コマンドライン引数からデータベース名を指定可能にする
default_restore_db_name="${default_db_name}_restore"

# バックアップファイルの選択
echo "以下のバックアップファイルから選択してください (キャンセルするには 'q' を入力してください):"
select backup_file in "$backup_dir"/*.sql; do
    if [ "$REPLY" == "q" ]; then
        echo "操作をキャンセルしました"
        exit 0
    elif [ -n "$backup_file" ]; then
        echo "選択されたバックアップファイル: $backup_file"
        break
    else
        echo "有効なファイルを選択するか、'q' を入力してキャンセルしてください。"
    fi
done

# 復元先のデータベース名を指定
read -p "復元先のデータベース名を入力してください (デフォルト: $default_restore_db_name): " new_db_name
new_db_name=${new_db_name:-"$default_restore_db_name"}

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
createdb -h "$default_db_host" -U "$default_db_user" "$new_db_name"

if [ $? -ne 0 ]; then
    echo "データベースの作成に失敗しました: $new_db_name"
    exit 1
fi

# 復元実行
echo "バックアップファイルからデータを復元しています..."
psql -h "$default_db_host" -U "$default_db_user" -d "$new_db_name" <"$backup_file"

if [ $? -eq 0 ]; then
    echo "復元が正常に完了しました: $new_db_name"
else
    echo "復元に失敗しました: $new_db_name"
    exit 1
fi
