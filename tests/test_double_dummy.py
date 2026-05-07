"""
双明手求解器测试

测试双明手求解器的基本功能：
1. 状态创建和深拷贝
2. 合法动作获取
3. 求解器基本功能
4. AI玩家包装器
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import random
from typing import List

from trick_taking.card import Card, Suit
from trick_taking.game_state import GameState
from trick_taking.utils.state_tools import create_random_state, create_state_from_hands
from trick_taking.solvers.double_dummy import DoubleDummySolver
from trick_taking.solvers import DoubleDummyPlayer
from trick_taking.games.spades import SpadesRules


class TestDoubleDummySolver(unittest.TestCase):
    """测试双明手求解器"""
    
    def setUp(self):
        """测试前准备"""
        self.solver = DoubleDummySolver(max_iterations=100)  # 减少迭代次数以加速测试
        self.rules = SpadesRules()
    
    def test_solver_initialization(self):
        """测试求解器初始化"""
        solver = DoubleDummySolver(max_iterations=1000, exploration_weight=1.4)
        self.assertEqual(solver.max_iterations, 1000)
        self.assertEqual(solver.exploration_weight, 1.4)
    
    def test_create_random_state(self):
        """测试创建随机状态"""
        state = create_random_state()
        
        # 验证基本属性
        self.assertEqual(len(state.hands), 4)
        for hand in state.hands:
            self.assertEqual(len(hand), 13)
        
        self.assertEqual(len(state.bids), 4)
        self.assertEqual(state.teams, [0, 1, 0, 1])
        self.assertEqual(state.phase, state.phase.PLAYING)
        self.assertEqual(state.tricks_played, 0)
    
    def test_state_legal_actions(self):
        """测试获取合法动作"""
        # 创建一个简单的测试状态
        state = create_random_state()
        current_player = state.turn
        
        # 获取合法动作
        hand = state.hands[current_player]
        legal_actions = self.rules.playable(state, hand, current_player)
        
        # 验证合法动作
        self.assertGreater(len(legal_actions), 0)
        for action in legal_actions:
            self.assertIsInstance(action, Card)
    
    def test_solver_single_action(self):
        """测试只有一个合法动作的情况"""
        # 创建一个状态，其中玩家只有一张牌
        from trick_taking.deck import Deck
        
        deck = Deck.standard_52()
        
        # 创建特殊手牌：玩家0只有一张黑桃A
        hands = [
            [Card(Suit.SPADES, "A")],  # 玩家0：只有一张牌
            deck.draw(13),  # 玩家1：正常手牌
            deck.draw(13),  # 玩家2：正常手牌
            deck.draw(13),  # 玩家3：正常手牌
        ]
        
        # 调整玩家1-3的手牌数量（因为抽走了黑桃A）
        for i in range(1, 4):
            hands[i] = hands[i][:13]
        
        bids = ["bid_1", "bid_1", "bid_1", "bid_1"]
        state = create_state_from_hands(hands, bids)
        
        # 求解
        result = self.solver.solve(state, current_player=0)
        
        # 验证结果
        self.assertIsNotNone(result["best_action"])
        self.assertEqual(result["best_action"], Card(Suit.SPADES, "A"))
    
    def test_state_deep_copy(self):
        """测试状态深拷贝"""
        state = create_random_state()
        
        # 创建求解器并测试深拷贝
        solver = DoubleDummySolver(max_iterations=10)
        copied_state = solver._deep_copy_state(state)
        
        # 验证深拷贝
        self.assertIsNot(state, copied_state)
        self.assertEqual(state.hands[0], copied_state.hands[0])
        
        # 修改拷贝，验证原状态不变
        if copied_state.hands[0]:
            original_first_card = state.hands[0][0]
            copied_state.hands[0].pop(0)
            self.assertIn(original_first_card, state.hands[0])
    
    def test_solver_empty_result(self):
        """测试空结果"""
        result = self.solver._empty_result(current_player=0)
        
        self.assertIsNone(result["best_action"])
        self.assertEqual(len(result["action_values"]), 0)
        self.assertEqual(result["state_evaluation"]["expected_score_diff"], 0.0)
        self.assertEqual(result["search_statistics"]["iterations"], 0)


class TestDoubleDummyPlayer(unittest.TestCase):
    """测试双明手AI玩家"""
    
    def setUp(self):
        """测试前准备"""
        self.player = DoubleDummyPlayer(max_iterations=50)
    
    def test_player_initialization(self):
        """测试玩家初始化"""
        player = DoubleDummyPlayer(max_iterations=500)
        self.assertEqual(player.solver.max_iterations, 500)
        self.assertEqual(player.position, 0)
        self.assertEqual(len(player.hand), 0)
        self.assertIsNone(player.full_state)
    
    def test_start_game(self):
        """测试游戏开始"""
        hand = [Card(Suit.HEARTS, "A"), Card(Suit.SPADES, "K")]
        self.player.start_game(position=2, hand=hand, num_players=4)
        
        self.assertEqual(self.player.position, 2)
        self.assertEqual(self.player.hand, hand)
    
    def test_place_bid(self):
        """测试叫牌"""
        # 测试有nil选项
        legal_bids = ["nil", "bid_1", "bid_2"]
        bid = self.player.place_bid(legal_bids, {})
        self.assertEqual(bid, "nil")
        
        # 测试无nil选项
        legal_bids = ["bid_1", "bid_2"]
        bid = self.player.place_bid(legal_bids, {})
        self.assertEqual(bid, "bid_1")
        
        # 测试空列表
        bid = self.player.place_bid([], {})
        self.assertIsNone(bid)


class TestStateTools(unittest.TestCase):
    """测试状态工具函数"""
    
    def test_state_to_dict_and_back(self):
        """测试状态字典转换"""
        state = create_random_state()
        
        # 转换为字典
        state_dict = create_state_from_hands.__globals__['state_to_dict'](state)
        
        # 验证字典结构
        self.assertIn("hands", state_dict)
        self.assertIn("bids", state_dict)
        self.assertIn("teams", state_dict)
        
        # 转换回状态
        restored_state = create_state_from_hands.__globals__['dict_to_state'](state_dict)
        
        # 验证恢复的状态
        self.assertEqual(len(restored_state.hands), 4)
        for i in range(4):
            self.assertEqual(
                [str(c) for c in restored_state.hands[i]],
                state_dict["hands"][i]
            )
    
    def test_compare_actions(self):
        """测试动作比较"""
        state = create_random_state()
        current_player = state.turn
        
        # 获取快速比较结果
        from trick_taking.utils.state_tools import compare_actions
        results = compare_actions(state, current_player, solver_iterations=10)
        
        # 验证结果
        self.assertGreater(len(results), 0)
        for action, info in results.items():
            self.assertIsInstance(action, Card)
            self.assertIn("value", info)
            self.assertIn("description", info)


def run_performance_test():
    """运行性能测试"""
    print("运行双明手求解器性能测试...")
    
    # 创建测试状态
    state = create_random_state()
    print(f"创建测试状态：玩家{state.turn}行动，剩余牌数={sum(len(h) for h in state.hands)}")
    
    # 测试不同迭代次数的性能
    iterations_list = [100, 500, 1000]
    
    for iterations in iterations_list:
        print(f"\n迭代次数: {iterations}")
        solver = DoubleDummySolver(max_iterations=iterations)
        
        import time
        start_time = time.time()
        result = solver.solve(state, current_player=state.turn)
        elapsed_time = time.time() - start_time
        
        print(f"  耗时: {elapsed_time:.2f}秒")
        print(f"  最优动作: {result['best_action']}")
        print(f"  扩展节点: {result['search_statistics']['nodes_expanded']}")
        
        if result['action_values']:
            top_action = result['action_values'][0]
            print(f"  最佳动作价值: {top_action['value']:.2f}")


def main():
    """主测试函数"""
    print("黑桃王双明手求解器测试")
    print("=" * 50)
    
    # 运行单元测试
    print("\n1. 运行单元测试...")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDoubleDummySolver)
    suite.addTests(loader.loadTestsFromTestCase(TestDoubleDummyPlayer))
    suite.addTests(loader.loadTestsFromTestCase(TestStateTools))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n单元测试结果: {result.testsRun}个测试, "
          f"{len(result.failures)}个失败, {len(result.errors)}个错误")
    
    # 运行性能测试
    print("\n2. 运行性能测试...")
    run_performance_test()
    
    # 示例使用
    print("\n3. 示例：使用双明手求解器")
    state = create_random_state()
    solver = DoubleDummySolver(max_iterations=200)
    result = solver.solve(state, current_player=state.turn)
    
    print(f"当前玩家: {state.turn}")
    print(f"最优动作: {result['best_action']}")
    
    if result['action_values']:
        print("\n前3个候选动作:")
        for i, action_info in enumerate(result['action_values'][:3]):
            print(f"  {i+1}. {action_info['action']}: "
                  f"价值={action_info['value']:.2f}, "
                  f"置信度={action_info['confidence']:.2f}")


if __name__ == "__main__":
    main()