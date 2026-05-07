"""
双明手AI玩家包装器

将DoubleDummySolver包装成AIPlayer接口，使其可以集成到现有的游戏循环中
"""

from __future__ import annotations

from typing import Any, List

from trick_taking.card import Card
from trick_taking.game_state import GameState
from trick_taking.player import AIPlayer

from .double_dummy import DoubleDummySolver


class DoubleDummyPlayer(AIPlayer):
    """
    双明手AI玩家
    
    这是一个包装器，将DoubleDummySolver包装成AIPlayer接口
    注意：此玩家需要访问完整的游戏状态（违反标准AIPlayer接口的限制）
    因此只适用于双明手分析场景
    """
    
    def __init__(self, max_iterations: int = 1000):
        """
        初始化双明手玩家
        
        参数:
            max_iterations: 最大搜索迭代次数
        """
        self.solver = DoubleDummySolver(max_iterations=max_iterations)
        self.position: int = 0
        self.hand: List[Card] = []
        self.full_state: GameState | None = None  # 完整游戏状态（双明手特有）
    
    def set_full_state(self, state: GameState) -> None:
        """
        设置完整游戏状态（双明手特有）
        
        标准AIPlayer接口不允许访问完整状态，但双明手求解需要
        此方法应在每次需要决策前调用
        """
        self.full_state = state
    
    # ─── AIPlayer接口实现 ──────────────────────────────────────────────
    
    def start_game(self, position: int, hand: List[Card], num_players: int) -> None:
        """游戏开始：记录位置和手牌"""
        self.position = position
        self.hand = hand.copy()
        print(f"双明手玩家{position}准备就绪，手牌数={len(hand)}")
    
    def place_bid(self, legal_bids: List[Any], state_view: dict) -> Any:
        """
        叫牌决策
        
        对于双明手求解器，我们假设叫牌已结束
        返回一个默认叫牌（实际游戏中应在叫牌阶段前使用）
        """
        # 双明手求解器主要针对叫牌后的出牌阶段
        # 这里返回一个合理的默认叫牌
        if "nil" in legal_bids:
            return "nil"
        elif "bid_1" in legal_bids:
            return "bid_1"
        elif legal_bids:
            return legal_bids[0]
        else:
            return None
    
    def play_card(self, legal_cards: List[Card], state_view: dict) -> Card:
        """
        出牌决策：使用双明手求解器选择最优出牌
        
        参数:
            legal_cards: 合法出牌列表
            state_view: 玩家视角的状态视图
            
        返回:
            最优出牌
        """
        if self.full_state is None:
            # 如果没有完整状态，随机选择
            print(f"警告：双明手玩家{self.position}无完整状态，随机出牌")
            return legal_cards[0] if legal_cards else None
        
        # 确保当前玩家是正确的
        if self.full_state.turn != self.position:
            print(f"警告：状态turn={self.full_state.turn}，玩家position={self.position}")
        
        # 使用双明手求解器求解
        result = self.solver.solve(self.full_state, self.position)
        
        best_action = result["best_action"]
        
        # 确保最优动作在合法动作中
        if best_action not in legal_cards:
            print(f"警告：最优动作{best_action}不在合法动作列表中")
            # 选择第一个合法动作作为备选
            best_action = legal_cards[0] if legal_cards else None
        
        if best_action:
            print(f"双明手玩家{self.position}出牌: {best_action}")
            
            # 打印部分评估信息
            action_values = result["action_values"]
            if action_values:
                top_actions = action_values[:3]
                print(f"  前3候选: " + ", ".join(
                    f"{a['action']}({a['value']:.1f})" for a in top_actions
                ))
        
        return best_action
    
    def bid_placed(self, player_id: int, bid_value: Any) -> None:
        """记录其他玩家的叫牌"""
        pass
    
    def set_teams(self, teams: List[int], bids: List[Any]) -> None:
        """设置队伍信息"""
        pass
    
    def card_played(self, player_id: int, card: Card) -> None:
        """记录其他玩家的出牌"""
        pass