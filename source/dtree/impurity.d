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
    import numir : size;
    return xs.sum!"fast" / xs.size;
}

auto variance(Xs)(Xs xs) {
    return sum!"fast"((xs - xs.mean) ^^ 2);
}
