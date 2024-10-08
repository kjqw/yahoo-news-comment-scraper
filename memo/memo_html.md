# メモ

暫定の記録。

## クラス名

- コメントの内容のタグ
  - dfmnYp
    - commentであるという意味の文字列
  - \<p class="sc-1mjevf1-11 dfmnYp"> のように、<sc-(ユーザーの識別子？)-(コメント欄の部分の番号) (commentであるという意味の文字列)> というクラス名が付与されている？
- 投稿時間
  - アクセスランキングページ
    - gNVspC
  - 記事ページ
    - knSIBn
      - 記事自体の投稿時間
    - eeXuVe
      - ページ内にある他の記事の投稿時間
  - コメントページ
    - iVnOtx

ここから調査を進める。上はテストとしてメモしたもの。

- 記事のコメントページ
  - 記事のタイトル
    - kvEDbM
    - aタグ
  - 記事の執筆者
    - cOuBjO
    - aタグ
  - 記事の配信時刻
    - p > span > time
    - クラス名はpタグのみについている
    - eetrFk
  - コメントの並び替え
    - 新着順
      - jLiYe
    - おすすめ順
      - vXGNJ
    - 現在選択されているほうはspanタグ、選択されていないほうはaタグ
    - つまり、変更にはページの遷移を伴う
    - urlに以下のようなパラメータを付ければ遷移する
      - ?order=newer
      - ?order=recommended
    - 標準はおすすめ順
  - コメントのユーザー名
    - kYZiqb
    - aタグ
  - コメントの投稿時間
    - cDKvmu
    - timeタグ
  - コメントの内容
    - dfmnYp
    - pタグ
  - コメントへの反応
    - 返信表示トグルボタン
      - hziPyv
      - buttonタグ
    - 返信数
      - eCOfVP
    - 「共感した」数
      - bGPsPv
      - spanタグ
    - 「なるほど」数
      - hxCIvY
      - spanタグ
    - 「うーん」数
      - kommN
      - spanタグ
    - 返信コメントのユーザー
      - jCzJt
      - リンクはeTqLVg
    - 返信コメントの投稿時間
      - jOWcmN
      - timeタグ
    - 返信コメントの内容
      - eRmkUe
      - pタグ
    - 返信コメントへの「共感した」数などのタグはコメントのものと同じ
    - 返信コメントを「もっと見る」ボタン
      - cKmtiL
      - buttonタグ
    - 返信コメントを「もっと見る」があと何件あるか
      - jjjDtk
      - spanタグ
  - 「次へ」ボタン
    - fxZmTZ
    - aタグ
    - これを押すと<https://news.yahoo.co.jp/articles/dc9de04285cfd432de93e6d8bee80eb7a41ef647/comments>から<https://news.yahoo.co.jp/articles/dc9de04285cfd432de93e6d8bee80eb7a41ef647/comments?page=2>のように遷移する
    - 「前へ」ボタンも同じクラス名
      - fxZmTZ
    - <data-cl-params="_cl_vmodule:page;_cl_link:pre;">だと「前へ」ボタン
    - <data-cl-params="_cl_vmodule:page;_cl_link:next;">だと「次へ」ボタン
    - 最初や最後のページには、「前へ」や「次へ」のaタグがない
  - 今表示されているコメントが何件から何件までか
    - kuObRn
    - emタグ
  - コメントが全部で何件あるか（返信を含まない？）
    - eFboGc
    - spanタグ
  - コメントが全部で何件あるか（返信を含む？）
    - jYUakM
    - spanタグ
- ユーザーのコメントページ
  - ユーザー名
    - hrCggG
    - pタグ
  - 記事
    - cFlsyG
      - pタグ
    - eebZSW
      - aタグ
  - コメント内容
    - edwfNw
    - pタグ
    - 記事に対するコメントでも、コメントに対する返信コメントでも、クラス名は同じ
  - コメントへの反応
    - eCOfVP
    - 「共感した」数
      - khvqxs
      - spanタグ
    - 「なるほど」数
      - dsHHTI
      - spanタグ
    - 「うーん」数
      - jPaIKS
      - spanタグ

## XPath

Copy full XPathで取得したものを記録。

- 記事のコメントページ
  - URLの例
    - <https://news.yahoo.co.jp/articles/97c7e42a75cb3293a06a14b2877dd4851f553f59/comments>
  - コメント部分のXPathと、タグのid
    - /html/body/div[1]/div/main/div[1]/div[1]/article
    - <article id="comment-main">
  - XPathと、取得できる情報の例
    - 記事のタイトル
      - /html/body/div[1]/div/main/div[1]/div[1]/article/header/h1/a
      - <a href="https://news.yahoo.co.jp/articles/97c7e42a75cb3293a06a14b2877dd4851f553f59" data-cl-params="_cl_vmodule:headline;_cl_link:title;" class="sc-aspdqx-2 bbtmYh" data-cl_cl_index="26">自民党総裁選「世代交代」求める動き…“２人の４０代議員”目指すものは？</a>
    - 記事の執筆者
      - /html/body/div[1]/div/main/div[1]/div[1]/article/header/p/a
      - <a href="/media/ann" data-cl-params="_cl_vmodule:headline;_cl_link:cp_name;" class="sc-aspdqx-4 bBnbRD" data-cl_cl_index="27">テレビ朝日系（ANN）</a>
    - 記事の配信時刻
      - /html/body/div[1]/div/main/div[1]/div[1]/article/header/p/span/time
      - <time>8/17(土)<!-- --> <!-- -->11:32</time>
    - コメントの並び替え
      <!-- - おすすめ順 -> 新着順
        - /html/body/div[1]/div/main/div[1]/div[1]/article/div[4]/ul/li[2]/a
        - <a href="comments?order=newer" data-cl-params="_cl_vmodule:cmtsort;_cl_link:post;_cl_position:0;" class="sc-1qgt3ce-3 jyBXnW" data-cl_cl_index="53">新着順</a>
      - 新着順 -> おすすめ順
        - /html/body/div[1]/div/main/div[1]/div[1]/article/div[4]/ul/li[1]/a
        - <a href="comments?order=recommended" data-cl-params="_cl_vmodule:cmtsort;_cl_link:empa;_cl_position:0;" class="sc-1qgt3ce-3 jyBXnW" data-cl_cl_index="30">おすすめ順</a> -->
      - 現在選択されている順番によってhtmlが変わるので、URLにパラメータを付けて遷移する方法のほうがわかりやすいかもしれない
        - おすすめ順
          - <https://news.yahoo.co.jp/.../comments>
          - <https://news.yahoo.co.jp/.../comments?order=recommended>
        - 新着順
          - <https://news.yahoo.co.jp/.../comments?order=newer>
    - 全コメント数
      - 1ページ目の場合
        - /html/body/div[1]/div/main/div[1]/div[1]/article/div[4]/div/p
      - それ以外の場合
        - /html/body/div[1]/div/main/div[1]/div[1]/article/div[2]/div/p
      - 例
        - <p class="sc-1qgt3ce-7 foYsJS">コメント<span class="sc-1qgt3ce-8 eHPJDs">2392</span>件</p>
    - エキスパートのコメント
      - コメントのユーザー名
        - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i]/article/div[2]/div[1]/h2/a
        - 例
          - <a href="https://news.yahoo.co.jp/profile/commentator/nakakitakoji/comments" data-cl-params="_cl_vmodule:cmt_pro;_cl_link:profnm;_cl_position:1;" class="sc-z8tf0-5 kBeXtL" data-cl_cl_index="55">中北浩爾</a>
          - <a href="https://news.yahoo.co.jp/profile/commentator/tanakayoshitsugu/comments" data-cl-params="_cl_vmodule:cmt_athr;_cl_link:profnm;_cl_position:2;" class="sc-z8tf0-5 kBeXtL" data-cl_cl_index="63">田中良紹</a>
      - コメントの投稿時間
        - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i]/article/div[2]/div[1]/time/a
        - 例
          - <a href="https://news.yahoo.co.jp/profile/commentator/nakakitakoji/comments/5f712c9b-3104-43d4-9892-9cc2ed327f88" data-cl-params="_cl_vmodule:cmt_pro;_cl_link:prmtime;_cl_position:1;" class="sc-z8tf0-8 czeFwq" data-cl_cl_index="56">5時間前</a>
          - <a href="https://news.yahoo.co.jp/profile/commentator/tanakayoshitsugu/comments/1bbc76fd-7b40-4401-b5b1-04095847273b" data-cl-params="_cl_vmodule:cmt_athr;_cl_link:prmtime;_cl_position:2;" class="sc-z8tf0-8 czeFwq" data-cl_cl_index="64">4時間前</a>
      - 投稿者の属性
        - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i]/article/div[2]/div[1]/p
        - 例
          - <p class="sc-z8tf0-9 kHCbXc">政治学者／中央大学法学部教授</p>
          - <p class="sc-z8tf0-9 kHCbXc">ジャーナリスト</p>
      - コメント内容を「もっと見る」ボタン
        - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i]/article/div[2]/p/button
        - これを押すことによって、コメントの内容が全て表示される
        - ボタンが存在しないこともある
      - コメントの内容
        - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i]/article/div[2]/p
        - 例
          - <p class="sc-z8tf0-11 bqbLoT"><a data-cl-params="_cl_vmodule:cmt_pro;_cl_link:com;_cl_position:1;" hidden="" data-cl_cl_index="58"></a><span class="sc-84c7oa-0 gAMNAA">解説</span>小泉氏と小林氏は、実に対照的です。小泉氏が総理を父親に持つ4代目の世襲であるのに対して、小林氏は商社マン家庭の出身。小泉氏が秘書経由でダイレクトに政界入りしたのに対して、小林氏は財務官僚から公募で政界入り。小泉氏が発信力に優れ、オーラを醸し出しているのに対して、小林氏は知的な議論を巧みに行い、謙虚な姿勢が特徴的です。小泉氏が無派閥を貫いているのに対して、小林氏は二階派に所属し、長老にも目をかけられてきました。政治信条的にも、小泉氏がライドシェアをはじめ規制緩和論者で、新自由主義的であるのに対して、小林氏は経済安全保障を推進するなど、国家の役割を重視する立場です。自民党の中堅・若手は仲良しクラブではなく、競争があるということに注目すべきでしょう。中堅・若手では、福田達夫・大野敬太郎・小倉将信の3氏が「自民党改革試案」を『文藝春秋』に発表しています。2段組み12ページの重厚な論文で、必読です。</p>
          - <p class="sc-z8tf0-11 bqbLoT"><a data-cl-params="_cl_vmodule:cmt_athr;_cl_link:vie;_cl_position:2;" hidden="" data-cl_cl_index="66"></a><span class="sc-84c7oa-3 idZAyQ">見解</span>小林鷹之氏を推す動きは「世代交代」とも見えるが、むしろ「小泉進次郎潰し」に私には見える。「世代交代」ならどちらか一人に乗れば良い訳で、小泉進次郎氏はかなり前から名前が挙がっていた。それは菅義偉氏や森喜朗氏が担ごうとしていたからだ。それに乗らずに小林鷹之氏を擁立するのは、「世代交代」より森喜朗、菅義偉氏らの影響力を排除しようとする中堅・若手の自民党改革だ。同様に麻生太郎氏の影響力も排除したいと中堅・若手は思っているのではないか。もはやキングメーカーは必要ないという総裁選になると思う。ただ政治は若ければ良いと言うわけではない。特に現在の世界情勢は米国の力の低下が半端ではない。これまでの日米同盟路線では立ち行かない状況だ。日本の世界戦略を巡る論戦も期待される。その意味では日本独立を主張する石破茂氏なども候補者となって論争が展開されることを望みたい。</p>
      - 参考になった数
        - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[1]/li[i]/article/div[2]/div[2]/button/span
        - 例
          - <span class="sc-x1bzsn-8 caCESi">2411</span>
          - <span class="sc-x1bzsn-8 caCESi">1272</span>
    - 一般人のコメント
      - コメントのユーザー名
        - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i]/article/article/div[2]/div[1]/h2/a
        - 例
          - <a href="https://news.yahoo.co.jp/users/UdLS1M4MKESzSPBMupvYF6EjKui7u1RnpXLkEvZXQtpRnFyQ00" data-cl-params="_cl_vmodule:cmt_usr;_cl_link:profnm;_cl_position:1;" class="sc-169yn8p-7 cJjfcA" data-cl_cl_index="76">lia*****</a>
          - <a href="https://news.yahoo.co.jp/users/wFxOa1V6RsqIpI-_k4wwji7jLA1JfDUikehJYE-PyQTAEsAv00" data-cl-params="_cl_vmodule:cmt_usr;_cl_link:profnm;_cl_position:2;" class="sc-169yn8p-7 cJjfcA" data-cl_cl_index="93">j_a********</a>
      - コメントの投稿時間
        - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i]/article/article/div[2]/div[1]/time/a
        - 例
          - <a href="https://news.yahoo.co.jp/profile/comments/289acf02-4f23-4e04-82fe-e55cfc22eceb" data-cl-params="_cl_vmodule:cmt_usr;_cl_link:prmtime;_cl_position:1;" class="sc-169yn8p-10 llygJK" data-cl_cl_index="75">6時間前</a>
          - <a href="https://news.yahoo.co.jp/profile/comments/635baa8d-42ae-4be9-a3da-8aff337c6b0b" data-cl-params="_cl_vmodule:cmt_usr;_cl_link:prmtime;_cl_position:2;" class="sc-169yn8p-10 llygJK" data-cl_cl_index="94">4時間前</a>
      - コメントの内容
        - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i]/article/article/div[2]/p
        - 例
          - <p class="sc-169yn8p-11 jeeyHa">世代交代すると新鮮なイメージになりますが、その新鮮なイメージの若手議員は、自民党の闇金問題の時に沈黙を守り動かず腐敗を正そうとしませんでした。骨抜きにされた政治資金規正法改正案に成立にも加担しました。新鮮なイメージになったところで、自民党は変わらず利権政治を続けることだけは確かです。</p>
          - <p class="sc-169yn8p-11 jeeyHa">高齢の方々が「昔はこうだった…」と新しい考えを取り入れず、時代についていけずにただ文句だけで居座り続けてもらうのも困りますが、「若い」ことがいいという短絡的な考えは危険です。実際、自分よりも若い上司の下についたこともありますが、頭がきれるなと思う反面、やはり経験不足や自信過剰なところがあり、他者の意見(特に反対意見や少数意見)を聞こうとしてくれず、非常に仕事がやりづらかったですし、その逆で高齢の上司でも、経験を活かして、職場環境を良くしてくださったこともあります。なので、年齢ではないなぁと私は思っています。人柄や素質など、様々な分野が総合的に関係してくるのでしょう。政治家でいえばやはり年齢も大切ですが、ビジョンかと思います。世代交代を謳うのであれば、どう世代交代をしていくのか？世代交代によってどんな変化が生まれるのか…それを前面に押し出していく必要があると思います。</p>
      - コメントに対する反応
        - 「共感した」数
          - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i]/article/article/div[2]/div[2]/div/ul/li[1]/button[2]/span/span[2]
          - 例
            - <span class="sc-1bswuwc-3 bGPsPv">6321</span>
            - <span class="sc-1bswuwc-3 bGPsPv">809</span>
        - 「なるほど」数
          - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i]/article/article/div[2]/div[2]/div/ul/li[2]/button[2]/span/span[2]
          - 例
            - <span class="sc-18v174d-3 hxCIvY">181</span>
            - <span class="sc-18v174d-3 hxCIvY">31</span>
        - 「うーん」数
          - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i]/article/article/div[2]/div[2]/div/ul/li[3]/button[2]/span/span[2]
          - 例
            - <span class="sc-1rqan7b-3 kommN">328</span>
            - <span class="sc-1rqan7b-3 kommN">77</span>
        - 返信を表示するボタン
          - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i]/article/article/div[2]/div[2]/button[1]
          - これを押すことによって、返信コメントのhtmlが追加される
          - 返信が0件でもこのボタンは存在する
          - 例
            - <button data-cl-params="_cl_vmodule:cmt_usr;_cl_link:opnre;_cl_position:1;" class="sc-169yn8p-13 fdgGrf" data-cl_cl_index="57"><svg height="18" width="18" class="sc-169yn8p-14 hwcxye riff-text-current" fill="currentColor" focusable="false" viewBox="0 0 48 48"><path clip-rule="evenodd" d="M35.947 16.739c-.017-.059-.029-.118-.056-.173-.026-.055-.064-.101-.101-.15-.042-.057-.082-.112-.135-.159-.015-.013-.021-.031-.038-.044-.033-.026-.074-.034-.11-.055a.993.993 0 0 0-.184-.093.921.921 0 0 0-.2-.04C35.081 16.019 35.044 16 35 16H13c-.044 0-.082.019-.124.025a.921.921 0 0 0-.2.04.954.954 0 0 0-.183.093c-.036.021-.077.029-.11.055-.017.013-.024.031-.039.044-.052.047-.092.102-.134.159-.037.049-.076.095-.102.15-.026.055-.039.114-.056.173-.018.068-.037.133-.041.203-.001.02-.011.037-.011.058 0 .043.019.081.024.123a.977.977 0 0 0 .041.199.971.971 0 0 0 .093.185c.021.036.029.077.056.11l11 14c.021.028.054.038.078.063.032.034.052.076.091.106.039.031.085.046.128.07.037.021.069.042.106.057A.994.994 0 0 0 24 32c.131 0 .259-.035.384-.087.037-.015.068-.036.103-.056.043-.025.09-.04.13-.071.039-.03.058-.072.091-.106.023-.025.056-.035.078-.063l11-14c.026-.033.034-.074.056-.11a.912.912 0 0 0 .092-.185.86.86 0 0 0 .041-.199c.005-.042.025-.08.025-.123 0-.021-.011-.038-.012-.058a.95.95 0 0 0-.041-.203Z" fill-rule="evenodd"></path></svg>返信<span class="sc-169yn8p-16 brKXoV">107</span>件</button>
        - 返信を「もっと見る」ボタン
          - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i]/div/div[1]/div/button
          - これを押すことによって、返信コメントのhtmlがさらに追加される
          - 表示できる返信コメントがなくなると、このボタンは消える
          - 例
            - <button data-cl-params="_cl_vmodule:cmtload;_cl_link:more;_cl_position:1;" class="sc-18ufaly-2 glDkuW" data-cl_cl_index="579"><svg height="20" width="20" class="sc-18ufaly-4 ktZwXI riff-text-current" fill="currentColor" focusable="false" viewBox="0 0 48 48"><path clip-rule="evenodd" d="M24 29.176 9.412 14.584a2.004 2.004 0 0 0-2.828 0 2.007 2.007 0 0 0 0 2.83l15.998 16.003c.39.39.904.584 1.418.583a1.994 1.994 0 0 0 1.418-.583l15.998-16.003a2.007 2.007 0 0 0 0-2.83 2.004 2.004 0 0 0-2.828 0L24 29.176Z" fill-rule="evenodd"></path></svg>もっと見る(<span class="sc-18ufaly-5 hsrbfV">87件</span>)</button>
        - 返信コメント
          - 返信コメントのユーザー名
            - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i]/div/ul/li[j]/div/article/div[2]/div[1]/h2/a
            - 例
              - <a href="https://news.yahoo.co.jp/users/aKkic1OdDJ36w6K7iY48tXpnuccgpMMAwvK5x4rF7jVo56w300" data-cl-params="_cl_vmodule:rep;_cl_link:profnm;_cl_position:1001;" class="sc-1v245k1-6 jrbpUu" data-cl_cl_index="321">tyr********</a>
          - 返信コメントの投稿時間
            - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i]/div/ul/li[j]/div/article/div[2]/div[1]/time
            - 例
              - <time class="sc-1v245k1-8 fPdGaj">5時間前</time>
          - 返信コメントの内容
            - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i]/div/ul/li[j]/div/article/div[2]/p
            - 例
              - <p class="sc-1v245k1-9 cenrGs">イメージで「信頼を取り戻す」作戦に出始めたと見て良いでしょう。おっしゃるとおり、福田氏はじめ、中堅若手は統一教会問題でも、裏金問題でも、自発的に動こうとはせず、上の顔色を伺ってきました。福田氏は統一教会の時には「何が問題なのか分からない」とまで発言していましたよね。裏金問題が出てきて更に自民党の信頼は失墜し、信頼を回復することよりも自分の次の選挙が心配になり、こうした動きをしているのだと思います。そんな人たちの綺麗事で顔をすげ替えても、自民党の本質は変わらない。</p>
          - 返信コメントに対する反応
            - 「共感した」数
              - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i]/div/ul/li[j]/div/article/div[2]/div[2]/ul/li[1]/button[2]/span/span[2]
              - 例
                - <<span class="sc-1bswuwc-3 bGPsPv">313</span>
            - 「なるほど」数
              - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i]/div/ul/li[j]/div/article/div[2]/div[2]/ul/li[2]/button[2]/span/span[2]
              - 例
                - <span class="sc-18v174d-3 hxCIvY">6</span>
            - 「うーん」数
              - /html/body/div[1]/div/main/div[1]/div[1]/article/ul[2]/li[i]/div/ul/li[j]/div/article/div[2]/div[2]/ul/li[3]/button[2]/span/span[2]
              - 例
                - <span class="sc-1rqan7b-3 kommN">21</span>
    - 現在のページのコメント数と、コメントの総数
      - /html/body/div[1]/div/main/div[1]/div[1]/article/div[6]/div/p
        - コメントページの1ページ目にはAI要約のdivがあるため、div[6]になっている
      - /html/body/div[1]/div/main/div[1]/div[1]/article/div[4]/div/p
        - 2ページ目以降にはAI要約のdivがないため、div[4]になっている
      - 例
        - <p class="sc-brfqoi-2 byUrZI"><em class="sc-12tq1dq-0 kuObRn">1〜10件</em>/<span class="sc-12tq1dq-1 eFboGc">1,939件</span></p>
    - ページ遷移
      - これはXPathで取得するのではなく、URLにパラメータを付けて遷移するほうがいいだろう
      - 例
        - <https://news.yahoo.co.jp/.../comments?page=2>
        - <https://news.yahoo.co.jp/.../comments?order=newer&page=3>
        - など

## メモ

- 「もっと見る」などを押すとhtmlが追加される。ボタンを押さないと返信コメントなどは取得できなそう？返信コメントの投稿時間などは、「もっと見る」を押さないで取得したhtmlには含まれていないが、押した後のhtmlには含まれていることを確認した。
- 文字はhタグではなくaタグで囲まれている？
  - aタグを取得したほうがいい？
- 投稿時間、反応数などはフォーマットがバラバラそうだから正規化が必要か
  - ...時間前、...月...日、...分前など
- ユーザー名を設定していない人の表示名は半角アルファベット3文字とアスタリスク8文字になっている？
  - 漢字、カタカナ、意味ありそうなアルファベットのユーザー名もある
- 返信コメントに返信はできない？
- kvEDbMなどのクラス名は、時間経過とともに変わる？
  - 2週間ほど経ってから再度アクセスしたらクラス名が変わっていた
  - XPathで取得するほうがいいかもしれない
