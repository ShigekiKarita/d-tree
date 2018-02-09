module dtree.impurity;

import mir.ndslice : map, slice;
import mir.math : sum, log;


auto gini(P)(P probs) {
    return 1.0 - (probs ^^ 2.0).sum!"fast";
}

auto entropy(P)(P probs) {
    if (probs.sum!"fast" == 0.0) return 0.0;
    return - probs.map!(p => p == 0.0 ? 0.0 : p * log(p)).sum!"fast";
}

auto mean(Xs)(Xs xs) {
    import numir : size;
    return xs.sum!"fast" / xs.size;
}

auto variance(Xs)(Xs xs) {
    return sum!"fast"((xs - xs.mean) ^^ 2);
}
