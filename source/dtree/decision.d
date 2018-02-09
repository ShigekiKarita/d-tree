module dtree.decision;

import std.stdio : writefln;

import mir.math : sum;
import mir.ndslice : ndarray, sliced, map, ipack, universal, slice, repeat, shape;
import numir : ones, zeros, zeros_like, Ndim, unsqueeze;


struct DecisionFields {
    size_t[] rightIndex, leftIndex;
    double[] leftPrediction, rightPrediction;
    double threshold;
    double impurity;
}

// TODO add tests
struct Regression {
    DecisionFields fields;
    alias fields this;

    void fit(X, Xs, Ys, I)(X x, Xs xs, Ys ys, I index, size_t fid, size_t nPreds) {
        threshold = x[fid];
        auto lrys = zeros(ys.length, ys[0].length);
        auto lmean = zeros_like(ys[0]);
        auto rmean = zeros_like(ys[0]);
        size_t li = 0, ri = ys.length-1;

        // TODO use mir-algorithm
        foreach (i; index) {
            if (xs[i][fid] > threshold) {
                rightIndex ~= [i];
                lrys[ri--][] = ys[i];
                rmean[] += ys[i];
            } else {
                leftIndex ~= [i];
                lrys[li++][] = ys[i];
                lmean[] += ys[i];
            }
        }
        // assert(li == ri);

        if (leftIndex.length > 0) lmean[] /= leftIndex.length;
        if (rightIndex.length > 0) rmean[] /= rightIndex.length;
        auto lys = lrys[0 .. li]; // ys[leftIndex.sliced];
        auto rys = lrys[li .. $]; // ys[rightIndex.sliced];
        for (size_t i = 0; i < ys[0].length; ++i) {
            lys[0 .. $, i] -= lmean[i];
            rys[0 .. $, i] -= rmean[i];
        }
        leftPrediction = lmean.ndarray;
        rightPrediction = rmean.ndarray;
        // FIXME need .slice in here iff compiling with dmd
        impurity = ((lys ^^ 2.0).slice.sum!"fast" + (rys ^^ 2.0).sum!"fast") / index.length;
    }
}


auto normalizeProb(T)(T probs) pure {
    import mir.math : sum;
    auto psum = probs.sum!"fast";
    return psum > 0.0 ? probs / psum : probs / 1.0;
}


// TODO add tests
struct Classification(alias ImpurityFun) {
    DecisionFields fields;
    alias fields this;

    void fit(X, Xs, Ys, I)(X x, Xs xs, Ys ys, I index, size_t fid, size_t nPreds) {
        threshold = x[fid];
        auto lpreds = zeros!double(nPreds);
        auto rpreds = zeros!double(nPreds);
        // TODO use mir-algorithm
        foreach (i; index) {
            if (xs[i][fid] > threshold) {
                rightIndex ~= [i];
                ++rpreds[ys[i]];
            } else {
                leftIndex ~= [i];
                ++lpreds[ys[i]];
            }
        }
        lpreds[] = lpreds.normalizeProb;
        rpreds[] = rpreds.normalizeProb;
        leftPrediction = lpreds.ndarray;
        rightPrediction = rpreds.ndarray;
        impurity = (ImpurityFun(lpreds) * leftIndex.length +
                    ImpurityFun(rpreds) * rightIndex.length) / xs.length;
    }
}

