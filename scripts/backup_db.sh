#!/bin/bash

# バックアップ先ディレクトリを指定
backup_dir="../db_backups"

# デフォルトのデータベース情報
default_db_host="postgresql_db"
default_db_user="kjqw"

# 実在するデータベースを取得
echo "データベース一覧を取得しています..."
databases=$(psql -h "$default_db_host" -U "$default_db_user" -d postgres -t -c "SELECT datname FROM pg_database WHERE datistemplate = false;" | sed '/^\s*$/d')

# データベースが取得できない場合は終了
if [ -z "$databases" ]; then
    echo "データベース一覧の取得に失敗しました。接続情報を確認してください。"
    exit 1
fi

# 配列に変換
available_dbs=($databases)

# バックアップ対象のデータベースを選択
echo "以下のデータベースからバックアップ対象を選択してください (キャンセルするには 'q' を入力してください):"
select db_name in "${available_dbs[@]}"; do
    if [ "$REPLY" == "q" ]; then
        echo "操作をキャンセルしました"
        exit 0
    elif [ -n "$db_name" ]; then
        echo "選択されたデータベース: $db_name"
        break
    else
        echo "有効なデータベースを選択するか、'q' を入力してキャンセルしてください。"
    fi
done

# 日付を付けたファイル名を作成
backup_file="${backup_dir}/$(date +%Y%m%d_%H%M%S)_${db_name}.sql"

# バックアップ先ディレクトリが存在しない場合は作成
if [ ! -d "$backup_dir" ]; then
    mkdir -p "$backup_dir"
fi

# バックアップ実行
pg_dump -h "$default_db_host" -U "$default_db_user" "$db_name" >"$backup_file"

# バックアップ成功の確認
if [ $? -eq 0 ]; then
    echo "バックアップが正常に完了しました: $backup_file"
else
    echo "バックアップに失敗しました: $db_name"
    exit 1
fi
