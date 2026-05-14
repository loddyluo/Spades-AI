"""黑桃王完整对局构造与运行脚本。

文件作用：
- 接收随机种子和四名玩家程序，随机发牌并进行完整一局黑桃王对局；
- 在每个出牌回合为当前玩家构造 1229 维输入特征，打印其输出动作，并校验合法性；
- 对每一步出牌后刷新四名玩家各自对应的 1229 维输入；
- 在终局后输出四名玩家得分；
- 额外支持随机生成多副牌，统计某个玩家的平均得分。

函数/类输入输出说明：
- build_random_state(seed: int) -> GameState
    输入: seed，随机数种子
    输出: 一个完成发牌、叫牌准备完成的 Spades 初始状态（52 张牌全部在手）

- SpadesMatchRunner.play_game() -> GameResult
    输入: runner 内部持有的规则、玩家列表和种子
    输出: 一局完整对局结果，包含 scores、tricks_won、winner

- play_random_match(seed: int, num_games: int, players_factory: callable) -> dict[str, float]
    输入: 初始种子、游戏数、玩家工厂函数
    输出: 平均得分统计字典

- main() -> None
    输入: 命令行参数
    输出: 打印单局演示和多局平均统计
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

sys.path.insert(0, ".")

from strategy.spades_player_programs import RandomSpadesPlayer, TruncatedMCTSPlayer
from strategy.truncated_mcts_strategy import TruncatedMCTSConfig
from trick_taking.card import Card
from trick_taking.deck import Deck, STANDARD_52
from trick_taking.game_state import Bid, GameState, Phase
from trick_taking.games.spades import SpadesRules
from trick_taking.utils.feature_encoder import SpadesFeatureEncoder


@dataclass
class PlayRecord:
    """单步出牌记录。"""

    player_id: int
    card: Card
    legal_cards: list[Card]
    feature_dim: int
    feature_snapshot: list[float] | None = None


def build_random_state(seed: int) -> GameState:
    """随机构造一个黑桃王初始牌局。

    输入:
    - seed: int，随机种子

    输出:
    - GameState: 四家手牌、叫牌和队伍都已初始化，等待进入对局。
    """
    rng = random.Random(seed)
    deck = Deck(STANDARD_52, seed=seed)
    hands = [deck.deal(13) for _ in range(4)]

    state = GameState()
    state.init_for_deal(4, hands, [], deck.all_cards)

    # 让起始牌局更稳定可复现：庄位与叫牌种子绑定。
    state.dealer_seat = rng.randrange(4)
    state.current_bidder = (state.dealer_seat + 1) % 4
    state.turn = state.current_bidder
    state.trick_leader = state.turn
    state.phase = Phase.BIDDING
    return state


class SpadesMatchRunner:
    """使用四个玩家程序驱动一整局黑桃王对局。"""

    def __init__(
        self,
        players: list,
        seed: int,
        verbose: bool = True,
        rules: SpadesRules | None = None,
        encoder: SpadesFeatureEncoder | None = None,
        on_card_played: Callable[[int, int], None] | None = None,
    ) -> None:
        if len(players) != 4:
            raise ValueError(f"需要 4 名玩家，实际得到 {len(players)} 名")
        self.players = players
        self.seed = seed
        self.verbose = verbose
        self.on_card_played = on_card_played
        # 默认采用”每人一次数值叫牌”的简化模式，避免 blind_nil/pass 造成同一玩家连续叫两次。
        self.rules = rules or SpadesRules(enable_nil=False, enable_blind_nil=False)
        self.encoder = encoder or SpadesFeatureEncoder()
        self.state = build_random_state(seed)
        self.player_features: dict[int, Any] = {}
        self.records: list[PlayRecord] = []

    def play_game(self):
        """运行一整局黑桃王，并返回 GameResult。"""
        self._start_game()
        self._bidding_phase()
        self._set_teams()
        self._play_phase()
        return self._score_game()

    def _start_game(self) -> None:
        """通知所有玩家开局并初始化各自手牌。"""
        for pid, player in enumerate(self.players):
            player.start_game(pid, list(self.state.hands[pid]), self.rules.num_players)
        self._refresh_all_player_features()

    def _bidding_phase(self) -> None:
        """执行叫牌阶段并打印每个玩家的输入维度。"""
        self.state.phase = Phase.BIDDING

        while not self.rules.end_bidding(self.state):
            bidder = self.state.current_bidder
            legal_bids = self.rules.legal_bids(self.state, bidder)
            view = self._build_view(bidder)
            bid = self.players[bidder].place_bid(legal_bids, view)
            if legal_bids and bid not in legal_bids:
                raise ValueError(f"玩家{bidder}叫牌非法: {bid!r}, 合法叫牌: {legal_bids}")

            self.state.bids.append(Bid(player_id=bidder, value=bid, is_pass=(bid == "pass")))
            if bid != "pass":
                self.state.max_bid[bidder] = bid

            if self.verbose:
                print(f"叫牌 | 玩家{bidder} | 输入维度={view['feature'].shape[0]} | 输出={bid}")

            for player in self.players:
                player.bid_placed(bidder, bid)

            self.state.current_bidder = self.rules.next_bid_turn(self.state)
            self._refresh_all_player_features()

    def _set_teams(self) -> None:
        """叫牌结束后设置队伍并通知所有玩家。"""
        self.state.teams = self.rules.set_team(self.state)
        self.state.points = self.rules.initial_points(self.state)
        bid_values = [bid.value for bid in self.state.bids]
        for player in self.players:
            player.set_teams(list(self.state.teams), bid_values)

    def _play_phase(self) -> None:
        """按回合进行出牌，直到 13 墩结束。"""
        self.state.phase = Phase.PLAYING
        trump_suits = self.rules.trump_mask(self.state)
        if trump_suits and len(trump_suits) == 1:
            self.state.trump_suit = next(iter(trump_suits))

        turn_count = 0
        trick_index = 0
        while not self.rules.end_trickgame(self.state):
            trick_index += 1
            if self.verbose:
                print(f"\n--- 第{trick_index:02d}墩开始 | 首攻玩家={self.state.turn} ---")
            self.state.table_cards = []
            for _ in range(self.rules.num_players):
                current = self.state.turn
                legal_cards = self.rules.playable(self.state, self.state.hands[current], current)
                if not legal_cards:
                    raise RuntimeError(f"玩家{current} 没有合法出牌")

                view = self._build_view(current)
                card = self.players[current].play_card(legal_cards, view)
                if card not in legal_cards:
                    raise ValueError(f"玩家{current} 出牌非法: {card}, 合法: {legal_cards}")

                if self.verbose:
                    print(
                        f"出牌 | 第{turn_count + 1:02d}步 | 玩家{current} | 输入维度={view['feature'].shape[0]} | 合法数={len(legal_cards)} | 输出={card}"
                    )

                self.state.play_card_to_table(current, card)
                if card.suit == self.state.trump_suit:
                    self.state.trump_broken = True
                    self.state.spades_broken = True

                for player in self.players:
                    player.card_played(current, card)

                self.records.append(
                    PlayRecord(
                        player_id=current,
                        card=card,
                        legal_cards=list(legal_cards),
                        feature_dim=int(view["feature"].shape[0]),
                        feature_snapshot=None,
                    )
                )

                self.state.turn = (current + 1) % self.rules.num_players
                turn_count += 1
                if self.on_card_played is not None:
                    self.on_card_played(turn_count, 52)

            winner = self.rules.winner_trick(self.state)
            self.state.complete_trick(winner)
            self.state.turn = winner
            self.state.trick_leader = winner

            if self.verbose:
                print(f"本墩赢家: 玩家{winner} | 下一墩首攻: 玩家{winner}")

            self._refresh_all_player_features()

        self.state.phase = Phase.SCORING

    def _score_game(self):
        """结算对局分数并返回 GameResult。"""
        from trick_taking.driver import GameResult

        scores = self.rules.score(self.state)
        self.state.points = scores
        winner = max(range(len(scores)), key=lambda i: scores[i])
        if self.verbose:
            print("\n终局得分：")
            for pid, score in enumerate(scores):
                print(f"  玩家{pid}: {score:.1f}")
            print(f"  本局赢家: 玩家{winner}")
        return GameResult(scores=scores, tricks_won=list(self.state.tricks_won), winner=winner, game_name=self.rules.game_name)

    def _build_view(self, player_id: int) -> dict[str, Any]:
        """构造当前玩家的 1229 维输入视图。"""
        feature = self.encoder.encode(self.state, player_id)
        view = copy.deepcopy(self.state.get_player_view(player_id))
        view["feature"] = feature
        view["state"] = copy.deepcopy(self.state)
        return view

    def _refresh_all_player_features(self) -> None:
        """刷新四名玩家各自当前的 1229 维输入。"""
        self.player_features = {
            pid: self.encoder.encode(self.state, pid) for pid in range(self.rules.num_players)
        }


def play_random_match(
    seed: int,
    num_games: int,
    players_factory: Callable[[int], list],
    rules_factory: Callable[[], SpadesRules] | None = None,
    verbose_first_game: bool = True,
) -> dict[str, float]:
    """运行多局对局并统计平均得分。

    输入:
    - seed: 起始随机种子
    - num_games: 对局数
    - players_factory: 接收 game_seed 并返回 4 名玩家对象的工厂函数
    - rules_factory: 规则构造函数；为 None 时使用默认 SpadesRules()
    - verbose_first_game: 是否打印第一局的详细过程

    输出:
    - 字典，包含 player1_avg_score、team0_avg_score、team1_avg_score
    """
    total_scores = [0.0, 0.0, 0.0, 0.0]
    for game_index in range(num_games):
        game_seed = seed + game_index
        runner = SpadesMatchRunner(
            players=players_factory(game_seed),
            seed=game_seed,
            verbose=(verbose_first_game and game_index == 0),
            rules=rules_factory() if rules_factory is not None else None,
        )
        result = runner.play_game()
        for pid, score in enumerate(result.scores):
            total_scores[pid] += float(score)

    return {
        "player1_avg_score": total_scores[1] / num_games,
        "team0_avg_score": (total_scores[0] + total_scores[2]) / num_games,
        "team1_avg_score": (total_scores[1] + total_scores[3]) / num_games,
    }


def build_default_players(seed: int, checkpoint_path: str | None, exact_threshold: int, leaf_threshold: int, simulations_per_action: int) -> list:
    """构造默认玩家：玩家1使用 MCTS，其余玩家随机。

    输入:
    - seed: 基础随机种子
    - checkpoint_path: MLP 权重路径
    - exact_threshold / leaf_threshold / simulations_per_action: MCTS 参数

    输出:
    - 长度为 4 的玩家列表
    """
    mcts_config = TruncatedMCTSConfig(
        exact_threshold=exact_threshold,
        leaf_threshold=leaf_threshold,
        simulations_per_action=simulations_per_action,
        checkpoint_path=checkpoint_path,
    )
    return [
        RandomSpadesPlayer(seed=seed + 0),
        TruncatedMCTSPlayer(config=mcts_config),
        RandomSpadesPlayer(seed=seed + 2),
        RandomSpadesPlayer(seed=seed + 3),
    ]


def main() -> None:
    """命令行入口：先跑 1 局完整演示，再跑 10 局随机对局评估。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="基础随机种子")
    parser.add_argument("--checkpoint", type=str, default="./result/mlp_test_3.pth", help="MCTS 使用的 MLP 权重")
    parser.add_argument("--exact_threshold", type=int, default=30, help="剩余牌数 <= 该值时直接精确求解")
    parser.add_argument("--leaf_threshold", type=int, default=24, help="MCTS 搜索到该剩余牌数时接入 MLP")
    parser.add_argument("--simulations_per_action", type=int, default=5, help="每个根动作模拟次数")
    parser.add_argument("--num_eval_games", type=int, default=10, help="随机对局评估局数")
    parser.add_argument("--enable_nil", action="store_true", help="是否启用 nil 叫牌")
    parser.add_argument("--enable_blind_nil", action="store_true", help="是否启用 blind_nil 叫牌")
    args = parser.parse_args()

    rules = SpadesRules(enable_nil=args.enable_nil, enable_blind_nil=args.enable_blind_nil)

    print("=== 单局演示：从 52 张牌开始完整出牌 ===")
    runner = SpadesMatchRunner(
        players=build_default_players(
            seed=args.seed,
            checkpoint_path=args.checkpoint,
            exact_threshold=args.exact_threshold,
            leaf_threshold=args.leaf_threshold,
            simulations_per_action=args.simulations_per_action,
        ),
        seed=args.seed,
        verbose=True,
        rules=rules,
    )
    result = runner.play_game()
    print(f"单局得分: {result.scores}")

    print("\n=== 10 局随机对局评估：玩家1 = MCTS，其余 = 随机 ===")

    def factory(game_seed: int) -> list:
        return build_default_players(
            seed=game_seed,
            checkpoint_path=args.checkpoint,
            exact_threshold=args.exact_threshold,
            leaf_threshold=args.leaf_threshold,
            simulations_per_action=args.simulations_per_action,
        )

    summary = play_random_match(
        seed=args.seed,
        num_games=args.num_eval_games,
        players_factory=factory,
        rules_factory=lambda: SpadesRules(enable_nil=args.enable_nil, enable_blind_nil=args.enable_blind_nil),
        verbose_first_game=False,
    )
    print(f"玩家1平均得分: {summary['player1_avg_score']:.2f}")
    print(f"队伍0平均得分: {summary['team0_avg_score']:.2f}")
    print(f"队伍1平均得分: {summary['team1_avg_score']:.2f}")


if __name__ == "__main__":
    main()
