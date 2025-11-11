import random

def play_game(player1_prob, player2_prob):
    score1 = 0
    score2 = 0

    while True:
        if random.random() < player1_prob:
            score1 += 1
        else:
            score2 += 1

        if score1 >= 21 and (score1 - score2 >= 2):
            return 1  # Player A wins
        if score2 >= 21 and (score2 - score1 >= 2):
            return 2  # Player B wins
        if score1 >= 30:
            return 1
        if score2 >= 30:
            return 2

def simulate_match(player1_prob, player2_prob, match_count):
    player1_wins = 0
    player2_wins = 0

    for _ in range(match_count):
        match_winner = play_game(player1_prob, player2_prob)
        if match_winner == 1:
            player1_wins += 1
        else:
            player2_wins += 1

    return player1_wins, player2_wins

def main():
    player_a_prob = float(input("请输入运动员 A 的每球获胜概率(0~1)："))
    player_b_prob = 1 - player_a_prob

    if player_a_prob < 0 or player_a_prob > 1:
        print("概率必须在0到1之间！")
        return

    match_count = int(input("请输入模拟的场次（正整数）："))
    if match_count <= 0:
        print("场次必须为正整数！")
        return

    player_a_wins, player_b_wins = simulate_match(player_a_prob, player_b_prob, match_count)

    print(f"\n模拟比赛总次数： {match_count}")
    print(f"A 选手每球获胜概率： {player_a_prob:.2f}")
    print(f"B 选手每球获胜概率： {player_b_prob:.2f}")
    print(f"共模拟了 {match_count} 场比赛")
    print(f"选手 A 获胜 {player_a_wins} 场，占比 {player_a_wins / match_count * 100:.1f}%")
    print(f"选手 B 获胜 {player_b_wins} 场，占比 {player_b_wins / match_count * 100:.1f}%")

    # 模拟小丹与小伟的职业生涯交手记录（假设数据）
    total_career_matches = 40
    career_a_wins = 28
    career_b_wins = total_career_matches - career_a_wins

    print(f"\n经统计，小丹跟小伟 14 年职业生涯，共交手 {total_career_matches} 次，小丹以 {career_a_wins}:{career_b_wins} 遥遥领先。")
    print(f"其中，两人共交战整整 {match_count} 局：")
    print(f"小丹获胜 {player_a_wins} 局，占比 {player_a_wins / match_count * 100:.1f}%；")
    print(f"小伟获胜 {player_b_wins} 局，占比 {player_b_wins / match_count * 100:.1f}%；")

if __name__ == "__main__":
    main()
