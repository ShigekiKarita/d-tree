module dtree.impurity;

import mir.ndslice : map;
import mir.math : sum, log;


auto gini(P)(P probs) {
    return 1.0 - sum!"fast"(probs ^^ 2.0);
}

auto entropy(P)(P probs) {
    return -sum!"fast"(probs * probs.map!log);
}

auto mean(Xs)(Xs xs) {
    return xs.sum!"fast" / xs.size;
}

auto regressionVariance(L, R)(L ls, R rs) {
    return sum!"fast"((ls - ls.mean) ^^ 2 + (rs - rs.mean) ^^ 2);
}
