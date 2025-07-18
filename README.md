# obs_scheduler
一晩で多数の天体を観測する場合に、どの順で観測していくのが合理的かの判断を支援する。

まず、設定ファイルobs_schedule.ymlを自分の環境に合わせて修正する。BSC5カタログのデータベースを使用する場合はBSC5.dbを置く場所に合わせてbsc_db_path:を修正する。

次に、観測天体のリストobs_list.txtを編集して、スケジュールを希望する観測開始日時とスケジューリング戦略、天体のリストを作成する。

実行するには、 $ python obs_schedule_gem.py

しばらく待つと、考慮された観測の順番に天体のリストが標準出力に表示される。標準出力なので、そのままだとコンソール画面に表示される。ファイルに出力を希望する場合は、コマンド入力時に次のように希望ファイル名にリダイレクトすれば、テキストファイルで得られる。

$ python obs_schedule_gem.py > scheduled.lst


このコードの開発に当たっては多くをAIに支援してもらいました。chatGPT, Gemini, DeepSeek, Copilot（すべて無料版）に仕様やアルゴリズムを指示してコーディングを行ってもう。実行してエラーが出れば指摘し、修正という繰り返しです。4種類のAIを利用したした主な訳は、無料版だとこのような作業を繰り返しているとすぐに容量制限にかかって翌日まで待たされるので、そこからは別のAIに相談するということを行ったためです。その他、別のAIのやり方だとどうなるかを見るという理由もあります。 
