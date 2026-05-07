#include <cstdint>

extern "C" {

// 为动作排序提供简易启发式分值。
// 分值越大，动作越优先被搜索（MAX节点方向）。
void score_actions(
    const int32_t* suits,
    const int32_t* ranks,
    int32_t n,
    int32_t table_has_spade,
    int32_t is_lead,
    int32_t* out_scores
) {
    for (int32_t i = 0; i < n; ++i) {
        int32_t suit_bonus = (suits[i] == 0) ? 20 : 0;  // 黑桃优先
        int32_t rank_bonus = ranks[i];

        if (table_has_spade && suits[i] == 0) {
            suit_bonus += 10;
        }
        if (is_lead && suits[i] == 0) {
            suit_bonus += 5;
        }

        out_scores[i] = suit_bonus + rank_bonus;
    }
}

// 计算一墩赢家（返回 winner_pid）。
// suits: 0=SPADES,1=HEARTS,2=DIAMONDS,3=CLUBS
int32_t trick_winner_pid(
    const int32_t* pids,
    const int32_t* suits,
    const int32_t* ranks,
    int32_t n,
    int32_t lead_suit
) {
    // 先看是否有黑桃
    int32_t best_pid = pids[0];
    int32_t best_rank = -1;
    bool has_spade = false;

    for (int32_t i = 0; i < n; ++i) {
        if (suits[i] == 0) {
            if (!has_spade || ranks[i] > best_rank) {
                has_spade = true;
                best_rank = ranks[i];
                best_pid = pids[i];
            }
        }
    }

    if (has_spade) {
        return best_pid;
    }

    // 无黑桃：比较引牌花色
    best_rank = -1;
    for (int32_t i = 0; i < n; ++i) {
        if (suits[i] == lead_suit) {
            if (ranks[i] > best_rank) {
                best_rank = ranks[i];
                best_pid = pids[i];
            }
        }
    }
    return best_pid;
}

}
