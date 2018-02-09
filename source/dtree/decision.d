module dtree.decision;

import std.stdio : writefln;

import mir.math : sum;
import mir.ndslice : ndarray, sliced, map, ipack, universal, slice, repeat, shape;
import numir : ones, zeros, zeros_like, Ndim, unsqueeze;


/// infomation for left/right decision used in Decision and Node
struct DecisionInfo {
    size_t[] index;
    double[] prediction;
    double impurity;
}


/// regression desicion implementation used at Node
struct Regression {
    DecisionInfo left, right;
    double threshold;

    void fit(X, Xs, Ys, I)(X x, Xs xs, Ys ys, I index, size_t fid, size_t nPreds) {
        threshold = x[fid];
        auto lrys = zeros(ys.length, ys[0].length);
        auto lmean = zeros_like(ys[0]);
        auto rmean = zeros_like(ys[0]);
        size_t li = 0, ri = ys.length-1;

        // TODO use mir-algorithm
        foreach (i; index) {
            if (xs[i][fid] > threshold) {
                right.index ~= [i];
                lrys[ri--][] = ys[i];
                rmean[] += ys[i];
            } else {
                left.index ~= [i];
                lrys[li++][] = ys[i];
                lmean[] += ys[i];
            }
        }
        // assert(li == ri);

        if (left.index.length > 0) lmean[] /= left.index.length;
        if (right.index.length > 0) rmean[] /= right.index.length;
        auto lys = lrys[0 .. li]; // ys[leftIndex.sliced];
        auto rys = lrys[li .. $]; // ys[rightIndex.sliced];
        for (size_t i = 0; i < ys[0].length; ++i) {
            lys[0 .. $, i] -= lmean[i];
            rys[0 .. $, i] -= rmean[i];
        }
        left.prediction = lmean.ndarray;
        right.prediction = rmean.ndarray;
        left.impurity = (lys ^^ 2.0).sum!"fast";
        right.impurity = (rys ^^ 2.0).sum!"fast";
    }
}

///
unittest {
    import mir.ndslice;
    import numir;
    import dtree.impurity;
    import std.stdio;

    Regression c;
    auto xs = [-2.0, -1.0, 0.0, 1.0].sliced.unsqueeze!1;
    auto ys = [-1.0, 0.0, 1.0, 2.0].sliced.unsqueeze!1;
    c.fit(xs[2], xs, ys, iota(ys.length), 0, 2);

    assert(c.left.index == [0, 1, 2]);
    assert(c.right.index == [3]);
    assert(c.threshold == xs[2][0]);
    assert(c.left.prediction == [0]);  // mean(-1, 0, 1)
    assert(c.right.prediction == [2]); // mean(2)
    assert(c.left.impurity == 2.0);  // ((-1-0)^2+(0-0)^2+(1-0)^2
    assert(c.right.impurity == 0.0); // (2-2)^2
}

auto normalizeProb(T)(T probs) pure {
    import mir.math : sum;
    auto psum = probs.sum!"fast";
    return psum > 0.0 ? probs / psum : probs / 1.0;
}


/// classification desicion implementation used at Node
struct Classification(alias ImpurityFun) {
    DecisionInfo left, right;
    double threshold;

    void fit(X, Xs, Ys, I)(X x, Xs xs, Ys ys, I index, size_t fid, size_t nPreds) {
        threshold = x[fid];
        auto lpreds = zeros!double(nPreds);
        auto rpreds = zeros!double(nPreds);
        // TODO use mir-algorithm
        foreach (i; index) {
            if (xs[i][fid] > threshold) {
                right.index ~= [i];
                ++rpreds[ys[i]];
            } else {
                left.index ~= [i];
                ++lpreds[ys[i]];
            }
        }
        lpreds[] = lpreds.normalizeProb;
        rpreds[] = rpreds.normalizeProb;
        left.prediction = lpreds.ndarray;
        right.prediction = rpreds.ndarray;
        left.impurity = ImpurityFun(lpreds) * left.index.length;
        right.impurity = ImpurityFun(rpreds) * right.index.length;
    }
}

///
unittest {
    import mir.ndslice;
    import numir;
    import dtree.impurity;

    {
        Classification!entropy c;
        auto xs = [-2.0, -1.0, 0.0, 1.0].sliced.unsqueeze!1;
        auto ys = [0, 0, 0, 1].sliced;
        c.fit(xs[2], xs, ys, iota(ys.length), 0, 2);

        assert(c.left.index == [0, 1, 2]);
        assert(c.right.index == [3]);
        assert(c.threshold == xs[2][0]);
        assert(c.left.prediction == [1, 0]);
        assert(c.right.prediction == [0, 1]);
        assert(c.left.impurity == 0.0);
        assert(c.right.impurity == 0.0);
    }
    {
        Classification!gini c;
        auto xs = [-2.0, -1.0, 0.0, 1.0].sliced.unsqueeze!1;
        auto ys = [0, 0, 0, 1].sliced;
        c.fit(xs[2], xs, ys, iota(ys.length), 0, 2);

        assert(c.left.index == [0, 1, 2]);
        assert(c.right.index == [3]);
        assert(c.threshold == xs[2][0]);
        assert(c.left.prediction == [1, 0]);
        assert(c.right.prediction == [0, 1]);
        assert(c.left.impurity == 0.0);
        assert(c.right.impurity == 0.0);
    }
}
