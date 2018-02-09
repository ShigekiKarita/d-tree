module dtree.impurity;

import mir.ndslice : map;
import mir.math : sum, log;


auto gini(P)(P probs) {
    return 1.0 - sum!"fast"(probs ^^ 2.0);
}

auto entropy(P)(P probs) {
    if (probs.sum!"fast" == 0.0) return 0.0;
    return -sum!"fast"(probs.map!(p => p == 0.0 ? 0.0 : p * log(p)));
}

auto mean(Xs)(Xs xs) {
    import numir : size;
    return xs.sum!"fast" / xs.size;
}

auto variance(Xs)(Xs xs) {
    return sum!"fast"((xs - xs.mean) ^^ 2);
}
