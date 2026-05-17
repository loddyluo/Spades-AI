统计规范：统计/home/oier/wyy/Spades-AI/logs/matchup_trace_seed0_games800_20260517_111831.txt这个文件。

得墩统计在每一局的末尾，格式例如：  
    seat 0: our_mcts           4/6
  seat 1: go_rule_2          4/1
  seat 2: our_mcts           4/2
  seat 3: go_rule_2          1/1

设共有A=800副牌，则共有2A个定约，A个是our_mcts的，A个是go_rule_2的
对于our_mcts队伍，取出它们的定约（例如，一边6/4，一边3/4，这说明他们总共叫了8墩，打成了9墩）。按照叫墩数（此处为8）和得墩数（此处为9）分类，形成一个二维数组，输出这个二维数组。二维数组所有数的总和应该是A。
对于our_mcts队伍，取出其中的所有x/0字段，给出x的统计分布。
对于go_rule_2队伍，也做同样的事情
对于上面的内容，尝试使用python统计，并且画图。
