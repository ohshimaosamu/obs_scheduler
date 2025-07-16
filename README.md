# obs_scheduler
一晩で多数の天体を観測する場合に、どの順で観測していくのが合理的かの判断を支援する。
まず、設定ファイルobs_schedule.ymlを自分の環境に合わせて修正する。BSC5カタログのデータベースを使用する場合はBSC%.dbを置く場所に合わせてbsc_db_path:を修正する。
次に、観測天体のリストobs_list.txtを編集して、希望する観測天体のリストを作成する。
実行するには、 $ python obs_schedule_gem.py
